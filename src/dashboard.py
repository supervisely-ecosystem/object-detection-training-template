import os
import json
import errno
import inspect
from dotenv import load_dotenv
from typing import Literal, List, Dict, Optional, Any
from types import SimpleNamespace

import torch 
import supervisely as sly
from torchvision.transforms import ToTensor
from supervisely.app.content import StateJson
from supervisely.app.widgets import (
    Container, Card, Button, Progress, Text, RadioTable, 
    RadioTabs, InputNumber, Grid, GridPlot, Tabs, Checkbox,
    ProjectThumbnail, ClassesTable, TrainValSplits, Select, 
    Input, Field, Editor, TabsDynamic, BindedInputNumber, 
    AugmentationsWithTabs, Switch, Stepper, Empty
)
import src.sly_globals as g

OPTIMIZERS = {x:v for (x,v) in torch.optim.__dict__.items() if inspect.isclass(v)}
SCHEDULERS = {x:v for (x,v) in torch.optim.lr_scheduler.__dict__.items() if inspect.isclass(v) and issubclass(v, torch.optim.lr_scheduler.__dict__['_LRScheduler']) and x != '_LRScheduler'}
AUG_TEMPLATES = [
    {'label': 'Light', 'value':'aug_templates/light.json'},
    {'label': 'Light + corrupt', 'value':'aug_templates/light_corrupt.json'},
    {'label': 'Medium', 'value':'aug_templates/medium.json'},
    {'label': 'Medium + corrupt', 'value':'aug_templates/medium_corrupt.json'},
    {'label': 'Hard', 'value':'aug_templates/hard.json'},
    {'label': 'Hard + corrupt', 'value':'aug_templates/hard_corrupt.json'},
]

class TrainDashboard:
    def __init__(
            self, 
            model,
            plots_titles: List[str],
            pretrained_weights: Dict[str, List] = None,
            hyperparameters_categories: List[str] = ['general', 'checkpoints', 'optimizer', 'intervals', 'scheduler'],
            extra_hyperparams: Dict[str, List] = {},
            hyperparams_edit_mode: Literal['ui', 'raw', 'all'] = 'ui',
            show_augmentations_ui: bool = True,
            augmentation_templates: List[Dict[str, str]] = [],
            task_type: Literal['detection', 'segmentation'] = 'detection',
            download_batch_size: int = 100,
            loggers: List = [],
        ):
        """
        Easy configuritible training dashboard for NN training
        
        Parameters
        ----------
        model : str
            Neural network model
        plots_titles : list = []
            Plots names for metrics grid. 
        pretrained_weights : dict = None
            Dict of pretrained weights for model. It will be displaye as table in pretrained tab in model settings card.
            Example:
            {
                'columns': ['Name', 'Description', 'Path'],
                'rows': [
                    ['Unet', 'Vanilla Unet', '/mnt/weights/unet.pth'],
                    ['Unet-11', 'VGG16', '/mnt/weights/unet11.pth'], 
                    ['Unet-16', 'VGG11', '/mnt/weights/unet16.pth']
                ]
            }
        hyperparameters_categories : list[str] = ['general', 'checkpoints', 'optimizer', 'intervals', 'scheduler']
            Tabs titles to show in hyperparameters card
        extra_hyperparams : dict[list] = {}
            age of the person
        hyperparams_edit_mode : Literal['ui', 'raw', 'all'] = 'all'
            age of the person
        show_augmentations_ui : bool = True
            age of the person
        loggers : list = [sly.logger]
            list of loggers, which support log() method

        Methods
        -------
        train():
            Run training for your model. 
        """
        self.model = model
        self._pretrained_weights = pretrained_weights
        self._hyperparameters = {}
        self._hyperparameters_categories = hyperparameters_categories
        self._extra_hyperparams = extra_hyperparams
        self._hyperparams_edit_mode = hyperparams_edit_mode
        self._show_augmentations_ui = show_augmentations_ui
        self._augmentation_templates = augmentation_templates + AUG_TEMPLATES
        self._task_type = task_type
        self._download_batch_size = download_batch_size
        self.loggers = SimpleNamespace(**{logger.__class__.__name__:logger for logger in loggers})
        
        self._content = []

        # Input project card
        self._project_preview = ProjectThumbnail(g.project)
        self._progress_bar_download_data = Progress(hide_on_finish=False)
        self._progress_bar_download_data.hide()
        self._text_download_data = Text('Project has been successfully downloaded', status='success')
        self._text_download_data.hide()
        self._button_download_dataset = Button('Download')
        @self._button_download_dataset.click
        def download_data():
            try:
                if sly.fs.dir_exists(g.project_dir):
                    sly.logger.info('Project already exist.')
                else:
                    self._button_download_dataset.hide()
                    self._progress_bar_download_data.show()
                    sly.fs.mkdir(g.project_dir)
                    with self._progress_bar_download_data(message=f"Processing images...", total=g.project.items_count) as pbar:
                        sly.Project.download(
                            api=g.api, 
                            project_id=g.project_id, 
                            dest_dir=g.project_dir,
                            batch_size=self._download_batch_size,
                            progress_cb=pbar.update,
                            only_image_tags=False, 
                            save_image_info=True)
                    sly.logger.info('Project successfully downloaded.')
                g.project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
                self._progress_bar_download_data.hide()
                self._button_download_dataset.hide()
                self._text_download_data.show()
                self._stepper.set_active_step(2)
                self.toggle_cards(['classes_table_card',], enabled=True)
            except Exception as e:
                print(f'ERROR: {str(e)}')
                self._progress_bar_download_data.hide()
                self._button_download_dataset.show()
                self._text_download_data.set('Project download failed', status='error')
                self._text_download_data.show()
                self._stepper.set_active_step(1)
        self._project_card = Card(
            title="Input project", 
            description="This project will be used for training",
            content=Container([self._project_preview, self._progress_bar_download_data, self._text_download_data, self._button_download_dataset])
        )

        # Classes table card
        self._classes_table = ClassesTable(project_id=g.project.id)
        @self._classes_table.value_changed
        def classes_list_changed(classes):
            if len(classes) > 0:
                self._button_classes_table.enable()
            else:
                self._button_classes_table.disable()
        self._button_classes_table = Button('Use selected classes')
        self._classes_table_card = Card(
            title="Classes table",
            description="Select classes, that should be used for training. Training supports only classes of shape Rectangle, other shapes are transformed to it automatically.",
            content=Container([self._classes_table, self._button_classes_table]),
        )
        @self._button_classes_table.click
        def toggle_classes_card():
            if self._classes_table_card.is_disabled() is True:
                self.toggle_cards(['classes_table_card'], enabled=True)
                self.toggle_cards([
                    'train_test_splits_card', 
                    'model_settings_card', 
                    'augmentations_card', 
                    'hyperparameters_card'
                ], enabled=False)
                self._button_splits.disable()
                self._button_model_settings.disable()
                self._button_augmentations_card.disable()
                self._button_hparams_card.disable()
                self._run_training_button.disable()
                self._button_classes_table.text = 'Use selected classes'
                self._button_splits.text = 'Create splits'
                self._button_model_settings.text = 'Select model'
                self._button_augmentations_card.text = 'Use selected augmentations'
                self._button_hparams_card.text = 'Use selected hyperparameters'
                self._stepper.set_active_step(2)
            else:
                self._button_classes_table.text = 'Select other classes'
                self.toggle_cards(['classes_table_card'], enabled=False)
                self.toggle_cards(['train_test_splits_card'], enabled=True)
                self._button_splits.enable()
                self._stepper.set_active_step(3)

        # Train / Validation splits card
        self.is_labels_converted = False
        self._splits = TrainValSplits(project_id=g.project.id)
        self._button_splits = Button('Create splits')
        self._progress_bar_labels_conversion = Progress(hide_on_finish=True)
        self._progress_bar_labels_conversion.hide()
        self._unlabeled_images_selector = Select([
            Select.Item(value="keep", label="keep unlabeled images"),
            Select.Item(value="skip", label="ignore unlabeled images"),
        ])
        self._train_test_splits_card = Card(
            title="Train / Validation splits",
            description="Define how to split your data to train/val subsets",
            content=Container([
                self._splits, 
                Field(
                    title="What to do with unlabeled images", 
                    description="Sometimes unlabeled images may be used to reduce noise in predictions, sometimes it is a mistake in training data", 
                    content=self._unlabeled_images_selector),
                self._progress_bar_labels_conversion,
                self._button_splits,
                ]),
        )
        @self._button_splits.click
        def toggle_splits_card():
            if self._train_test_splits_card.is_disabled() is True:
                self.toggle_cards(['train_test_splits_card'], enabled=True)
                self.toggle_cards([
                    'model_settings_card', 
                    'augmentations_card', 
                    'hyperparameters_card'
                ], enabled=False)
                self._button_splits.text = 'Create splits'
                self._button_model_settings.text = 'Select model'
                self._button_augmentations_card.text = 'Use selected augmentations'
                self._button_hparams_card.text = 'Use selected hyperparameters'
                self._button_model_settings.disable()
                self._button_augmentations_card.disable()
                self._button_hparams_card.disable()
                self._run_training_button.disable()
                self._stepper.set_active_step(3)
            else:
                self.toggle_cards(['train_test_splits_card',], enabled=False)
                self.toggle_cards(['model_settings_card',], enabled=True)
                self._button_splits.text = 'Recreate splits'
                self._button_model_settings.enable()
                self._progress_bar_labels_conversion.show()
                if not self.is_labels_converted:
                    with self._progress_bar_labels_conversion(message=f"Converting labels for {self._task_type} task", total=g.project.items_count) as pbar:
                        if self._task_type == 'detection':
                            sly.Project.to_detection_task(
                                src_project_dir=g.project_dir, 
                                inplace=True, 
                                progress_cb=pbar.update)
                        elif self._task_type == 'segmentation':
                            sly.Project.to_segmentation_task(
                                src_project_dir=g.project_dir, 
                                inplace=True, 
                                target_classes=self._classes_table.get_selected_classes(),
                                progress_cb=pbar.update)
                    self.is_labels_converted = True
                self._stepper.set_active_step(4)

        # Model settings card
        self._weights_path_input = Input(placeholder="Path to .pt file in Team Files")
        titles = ["Scratch", "Custom weights"]
        descriptions = ["Model training from scratch", "",]
        contents = [Text("Training from scratch", status="info"), self._weights_path_input]
        self._progress_bar_download_model = Progress(hide_on_finish=True)
        self._text_download_model = Text('Model has been successfully downloaded', status='success')
        self._text_download_model.hide()
        self._progress_bar_download_model.hide()
        if self._pretrained_weights:
            self._weights_table = RadioTable(**self._pretrained_weights)
            titles.insert(1, "Pretrained")
            descriptions.insert(1, "Model pretrained checkpoints")
            contents.insert(1, self._weights_table)
        else:
            self._weights_table = Empty()
        self._model_settings_tabs = RadioTabs(titles, contents, descriptions)
        self._button_model_settings = Button('Select model')
        self._model_settings_card = Card(
            title="Model settings",
            description="Choose model size or how weights should be initialized",
            content=Container([self._model_settings_tabs, self._progress_bar_download_model, self._text_download_model, self._button_model_settings]),
        )
        @self._button_model_settings.click
        def toggle_model_settings_card():
            if self._model_settings_card.is_disabled() is True:
                self.toggle_cards(['model_settings_card'], enabled=True)
                self.toggle_cards([
                    'augmentations_card', 
                    'hyperparameters_card'
                ], enabled=False)
                self._button_model_settings.text = 'Select model'
                self._button_augmentations_card.text = 'Use selected augmentations'
                self._button_hparams_card.text = 'Use selected hyperparameters'
                self._button_augmentations_card.disable()
                self._button_hparams_card.disable()
                self._run_training_button.disable()
                self._text_download_model.hide()
                self._stepper.set_active_step(4)
            else:
                self.toggle_cards(['model_settings_card',], enabled=False)
                self.toggle_cards(['augmentations_card',], enabled=True)
                if not self._show_augmentations_ui:
                    self.toggle_cards(['hyperparameters_card',], enabled=True)
                    self._button_hparams_card.enable()
                self._button_model_settings.text = 'Change model'
                self._button_augmentations_card.enable()
                
                self.pretrained_weights_path = self.get_pretrained_weights_path()
                if self.pretrained_weights_path is not None:
                    full_path = os.path.normpath(g.checkpoints_dir + self.pretrained_weights_path)
                    if not os.path.exists(full_path):
                        self._progress_bar_download_model.show()
                        self.download_sly_file(
                            remote_path=self.pretrained_weights_path, 
                            local_path=full_path, 
                            progress=self._progress_bar_download_model
                        )
                        self._text_download_model.show()
                        sly.logger.info('Model weights successfully downloaded.')
                    else:
                        self._text_download_model.show()
                        sly.logger.info('Model weights already exists.')
                self._stepper.set_active_step(5)

        # Training progress card
        self._run_training_button = Button('Start training')
        @self._run_training_button.click
        def run_training():
            self.toggle_cards([
                    'classes_table_card',
                    'train_test_splits_card', 
                    'model_settings_card', 
                    'augmentations_card', 
                    'hyperparameters_card'
            ], enabled=False)
            self._button_classes_table.disable()
            self._button_splits.disable()
            self._button_model_settings.disable()
            self._button_augmentations_card.disable()
            self._button_hparams_card.disable()
            self._run_training_button.disable()
            self._run_training_button.disable()
            try:
                self.progress_bar.show()
                if 'SummaryWriter' in vars(self.loggers):
                    from tensorboard import program
                    log_dir = vars(self.loggers)['SummaryWriter'].log_dir
                    tb = program.TensorBoard()
                    tb.configure(argv=[None, '--logdir', log_dir])
                    url = tb.launch()
                    print(f"Tensorflow listening on {url}")
                self.train()
            except Exception as e:
                self._button_classes_table.enable()
                self._button_splits.enable()
                self._button_model_settings.enable()
                self._button_augmentations_card.enable()
                self._button_hparams_card.enable()
                self._run_training_button.enable()
                raise e

        self.progress_bar = Progress(hide_on_finish=False)
        self.progress_bar.hide()
        self._logs_editor = Editor(
            initial_text='', 
            height_px=250,
            language_mode='plain_text', 
            restore_default_button=False, 
            readonly=True, 
            highlight_active_line=False,
            show_line_numbers=False)
        self._grid_plot = GridPlot(plots_titles, columns=3)
        self._logs_card = Card(title='Logs', content=self._logs_editor, collapsable=True)
        self._grid_plot_card = Card(title='Metrics', content=self._grid_plot, collapsable=True)
        self._training_card = Card(
            title="Training progress",
            description="Task progress, detailed logs, metrics charts, and other visualizations",
            content=Container([self._run_training_button, self.progress_bar, self._logs_card, self._grid_plot_card]),
        )
        
        self._content += [
            self._project_card,
            self._classes_table_card,
            self._train_test_splits_card,
            self._model_settings_card
        ]

        # Training augmentations card
        if self._show_augmentations_ui:
            self._switcher_augmentations = Switch(switched=True)
            self._augmentations = AugmentationsWithTabs(
                globals=g,
                templates=self._augmentation_templates, 
                task_type=self._task_type
            )
            @self._switcher_augmentations.value_changed
            def augs_switcher_toggle(val):
                if val:
                    self._augmentations.show()
                else:
                    self._augmentations.hide()
            self._button_augmentations_card = Button('Use selected augmentations')
            self._augmentations_card = Card(
                title="Training augmentations",
                description="Choose one of the prepared templates or provide custom pipeline",
                content=Container([
                    Field(title='Augmentations', content=self._switcher_augmentations), 
                    self._augmentations, 
                    self._button_augmentations_card
                ]),
            )
            @self._button_augmentations_card.click
            def toggle_augmentations_card():
                if self._augmentations_card.is_disabled() is True:
                    self.toggle_cards(['augmentations_card'], enabled=True)
                    self.toggle_cards(['hyperparameters_card'], enabled=False)
                    self._button_augmentations_card.text = 'Use selected augmentations'
                    self._button_hparams_card.text = 'Use selected hyperparameters'
                    self._button_hparams_card.disable()
                    self._run_training_button.disable()
                    self._stepper.set_active_step(5)
                else:
                    self.toggle_cards(['augmentations_card',], enabled=False)
                    self.toggle_cards(['hyperparameters_card',], enabled=True)
                    self._button_augmentations_card.text = 'Change augmentations'
                    self._button_hparams_card.enable()
                    self._run_training_button.disable()
                    self._stepper.set_active_step(6)
            self._content.append(self._augmentations_card)
        else:
            self._augmentations = Empty()           
            self._augmentations_card = Empty()  
            self._switcher_augmentations = Empty()  
            self._button_augmentations_card = Empty()  

        self._content += [self.hyperparameters_card(), self._training_card]
        self._stepper = Stepper(widgets=self._content)

        self.toggle_cards([
            'classes_table_card',
            'train_test_splits_card',
            'model_settings_card',
            'augmentations_card',
            'hyperparameters_card',
        ], enabled=False)
        self._button_classes_table.disable()
        self._button_splits.disable()
        self._button_model_settings.disable()
        self._button_augmentations_card.disable()
        self._button_hparams_card.disable()
        self._run_training_button.disable()

    def get_pretrained_weights_path(self):
        selected_trainig_mode = self._model_settings_tabs.get_active_tab()
        if selected_trainig_mode == 'Pretrained':
            path_col_index = self._weights_table.columns.index('Path')
            weights_path = self._weights_table.get_selected_row(StateJson())[path_col_index]
        elif selected_trainig_mode == 'Custom weights':
            weights_path = self._weights_path_input.get_value()
        else:
            weights_path = None
        return weights_path

    def get_hyperparameters(self):
        hparams_from_ui = {}
        for tab_label, param in self._hyperparameters.items():
            hparams_from_ui[tab_label] = {}
            for key, widget in param.items():
                if isinstance(widget, Checkbox):
                    hparams_from_ui[tab_label][key] = widget.is_checked()
            for key, widget in param.items():
                if isinstance(widget, Switch):
                    hparams_from_ui[tab_label][key] = widget.is_switched()
                else:
                    hparams_from_ui[tab_label][key] = widget.get_value()

        if self._hyperparams_edit_mode in ['raw', 'all']:
            hparams_from_file = self._hyperparameters_tab_dynamic.get_merged_yaml(as_dict=True)
            # converting OrderedDict to simple dict
            hparams_from_file = json.loads(json.dumps(hparams_from_file))
            for key, value in hparams_from_ui.items():
                if key in hparams_from_file.keys():
                    hparams_from_file[key].update(value)
                else:
                    hparams_from_file[key] = value
            return hparams_from_file
        else:
            return hparams_from_ui
    
    def hyperparameters_ui(self):
        hparams_widgets = {}
        if 'general' in self._hyperparameters_categories:
            hparams_widgets['general'] = [
                dict(key='number_of_epochs',
                    title='Number of epochs', 
                    description='Total count epochs for training', 
                    content=InputNumber(10, min=1, max=100000, size='small')),
                dict(key='batch_size',
                    title='Batch size', 
                    description='total batch size for all GPUs. Use the largest batch size your GPU allows. For example: 16 / 24 / 40 / 64 (batch sizes shown for 16 GB devices)', 
                    content=InputNumber(8, min=6, max=100000, size='small')),
                dict(key='input_image_size',
                    title='Input image size (in pixels)', 
                    description='Image is resized to square', 
                    content=BindedInputNumber(width=256, height=256, max=1024)),
                dict(key='device',
                    title='Device', 
                    description='Cuda device, i.e. 0 or 0,1,2,3 or cpu, or keep empty to select automatically', 
                    content=Input('0', size='small')),
                dict(key='workers_number',
                    title='Number of workers', 
                    description='Maximum number of dataloader workers, use 0 for debug', 
                    content=InputNumber(8, min=0, size='small')),
                *self._extra_hyperparams.get('general', [])
            ]
        if 'optimizer' in self._hyperparameters_categories:
            hparams_widgets['optimizer'] = [
                dict(key='name',
                    title='Optimizer', 
                    description='Setup corresponding learning rate for Adam in additional configuration, default values are provided for SGD', 
                    content=Select([Select.Item(value=key, label=key) for key in OPTIMIZERS.keys()], size='small')),
                dict(key='lr',
                    title='Learning rate', 
                    description='',
                    content=InputNumber(0.001, min=1e-10, max=1e10, step=0.001, size='small')),
                dict(key='foreach',
                    title='Foreach', 
                    description='Whether foreach implementation of optimizer is used',
                    content=Switch(switched=False)),
                dict(key='maximize',
                    title='Maximize', 
                    description='Maximize the params based on the objective, instead of minimizing',
                    content=InputNumber(0.0, min=0, max=1.0, step=0.1, size='small')),
                dict(key='eps',
                    title='Eps', 
                    description='Term added to the denominator to improve numerical stability',
                    content=InputNumber(1e-8, min=1e-16, max=1e16, step=0.1, size='small')),
                *self._extra_hyperparams.get('optimizer', [])
        ]
        if 'checkpoints' in self._hyperparameters_categories:
            hparams_widgets['checkpoints'] = [
                *self._extra_hyperparams.get('checkpoints', [])
            ]
        if 'intervals' in self._hyperparameters_categories:
            hparams_widgets['intervals'] = [
                dict(key='logging',
                    title='Logging interval', 
                    description='How often metrics should be logged, increase if training data is small', 
                    content=InputNumber(1, min=1, max=10000, size='small')),
                dict(key='validation',
                    title='Validation interval', 
                    description='How often to estimate the model on validation dataset', 
                    content=InputNumber(10, min=1, max=10000, size='small')),
                dict(key='сheckpoints',
                    title='Checkpoints interval', 
                    description='How often to save the model weights', 
                    content=InputNumber(100, min=1, max=10000, size='small')),
                *self._extra_hyperparams.get('intervals', [])
            ]
        if 'scheduler' in self._hyperparameters_categories:            
            hparams_widgets['scheduler'] = [
                dict(key='name',
                    title='Scheduler', 
                    description='Learning rate scheduler', 
                    content=Select([Select.Item(value=key, label=key) for key in SCHEDULERS.keys()], size='small')),
                *self._extra_hyperparams.get('scheduler', [])
            ]

        return hparams_widgets 

    def hyperparameters_card(self):
        card_content = []

        labels = []
        contents = []
        if self._hyperparams_edit_mode in ('ui', 'all'):
            for tab_label, widgets in self.hyperparameters_ui().items():
                self._hyperparameters[tab_label] = {}
                grid_content = []
                if len(widgets) == 0:
                    continue
                for hparam in widgets:
                    grid_content.append(Field(hparam['content'], hparam['title'], hparam['description']))
                    self._hyperparameters[tab_label][hparam['key']] = hparam['content']
                hyperparameters_grid = Grid(grid_content, columns=1)
                labels.append(tab_label.capitalize())
                contents.append(hyperparameters_grid)
            hparams_tabs = Tabs(labels, contents)
            self._hyperparameters_tab_dynamic = Empty()
            card_content.append(hparams_tabs)
        if self._hyperparams_edit_mode in ('raw', 'all'):
            # self._hyperparameters_file_selector = Select([
            #     Select.Item(value=t['value'], label=t['label']) for t in self._hyperparams_templates
            # ])
            # @self._hyperparameters_file_selector.value_changed
            # def hyperparams_file_changed(value):
            #     print(f"New file is: {value}")
            #     # TODO
            #     # self._hyperparameters_tab_dynamic.reinit(value)
            # hyperparameters_selector_field = Field(title="Hyperparameters file", 
            #                     description="Choose from provided files or select own from team files", 
            #                     content=self._hyperparameters_file_selector)
            # self._hyperparameters_tab_dynamic = TabsDynamic(self._hyperparameters_file_selector.get_value())

            raw_yaml = """# example
            sample_param: 0.0003
            """
            self._hyperparameters_tab_dynamic = TabsDynamic(raw_yaml)
            hyperparameters_field = Field(title="Additional hyperparameters", 
                                          description="Define addition hyperparameters as YAML.", 
                                          content=self._hyperparameters_tab_dynamic)
            card_content += [
                # hyperparameters_selector_field, 
                hyperparameters_field
            ]
        
        self._button_hparams_card = Button('Use selected hyperparameters')
        card_content += [self._button_hparams_card]
        self._hyperparameters_card = Card(
            title="Traning hyperparameters",
            description="Define general settings and advanced configuration (learning rate, augmentations, ...)",
            content=Container(card_content)
        )
        @self._button_hparams_card.click
        def toggle_hparams_card():
            if self._hyperparameters_card.is_disabled() is True:
                self._button_hparams_card.text = 'Use selected hyperparameters'
                self.toggle_cards(['hyperparameters_card', 'augmentations_card'], enabled=True)
                self._run_training_button.disable()
                if self._show_augmentations_ui:
                    self._stepper.set_active_step(6)
                else:
                    self._stepper.set_active_step(5)
            else:
                self._button_hparams_card.text = 'Change hyperparameters'
                self.toggle_cards(['hyperparameters_card'], enabled=False)
                self._run_training_button.enable()
                if self._show_augmentations_ui:
                    self._stepper.set_active_step(7)
                else:
                    self._stepper.set_active_step(6)
        return self._hyperparameters_card

    def get_splits(self):
        self._splits._project_fs = g.project_fs
        return self._splits.get_splits()

    def get_optimizer(self, name: str):
        return torch.optim.__dict__[name]

    def get_scheduler(self, name: str):
        return torch.optim.lr_scheduler.__dict__[name]

    def get_transforms(self):
        if self._switcher_augmentations.is_switched():
            augs_pipeline, augs_py_code = self._augmentations.get_augmentations()
            return augs_pipeline
        else:
            return None

    def train(self):
        raise NotImplementedError('You need to define train loop for your model')

    def log(self, method: str, **kwargs):
        for logger_name, logger in vars(self.loggers).items():
            func = getattr(logger, method)
            func(**kwargs)
        
        if method in ('add_scalars'):
            self._grid_plot.add_scalar(kwargs['tag'], kwargs['scalar_value'], kwargs['global_step'])

        if method in ('add_text'):
            sly.logger.info(kwargs['text_string'])
            self._logs_editor.set_text(self._logs_editor.get_text() + f"\n{kwargs['text_string']}")
        
    def toggle_cards(self, cards, enabled: bool = False):
        if 'classes_table_card' in cards:
            if enabled:
                self._classes_table_card.enable()
                self._classes_table.enable()
            else:
                self._classes_table_card.disable()
                self._classes_table.disable()

        if 'train_test_splits_card' in cards:
            if enabled:
                self._train_test_splits_card.enable()
                self._splits.enable()
                self._unlabeled_images_selector.enable()
            else:
                self._train_test_splits_card.disable()
                self._splits.disable()
                self._unlabeled_images_selector.disable()

        if 'model_settings_card' in cards:
            if enabled:
                self._model_settings_card.enable()
                self._weights_path_input.enable()
                self._weights_table.enable()
                self._model_settings_tabs.enable()
            else:
                self._model_settings_card.disable()
                self._weights_path_input.disable()
                self._weights_table.disable()
                self._model_settings_tabs.disable()
        
        if 'augmentations_card' in cards:
            if enabled:
                self._augmentations_card.enable()
                self._switcher_augmentations.enable()
                self._augmentations.enable()
            else:
                self._augmentations_card.disable()
                self._switcher_augmentations.disable()
                self._augmentations.disable()

        if 'hyperparameters_card' in cards:
            if enabled:
                self._hyperparameters_card.enable()
                self._hyperparameters_tab_dynamic.enable()
                for tab_label, param in self._hyperparameters.items():
                    for key, widget in param.items():
                        widget.enable()
            else:
                self._hyperparameters_card.disable()
                self._hyperparameters_tab_dynamic.disable()
                for tab_label, param in self._hyperparameters.items():
                    for key, widget in param.items():
                        widget.disable()

    def download_sly_file(self, remote_path, local_path, progress = None):
        file_info = g.api.file.get_info_by_path(g.team.id, remote_path)
        if file_info is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), remote_path)

        with progress(message=f"Download model weights..", total=file_info.sizeb) as pbar:
            g.api.file.download(g.team.id, remote_path, local_path, progress_cb=pbar.update)
        sly.logger.info(f"{remote_path} has been successfully downloaded", extra={"weights": local_path})

    def run(self):
        return sly.Application(
            layout=self._stepper
        )