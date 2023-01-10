import os
import json
import inspect
from dotenv import load_dotenv
from typing import Literal

import torch 
import supervisely as sly
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from supervisely.app.content import DataJson, StateJson
from supervisely.app.widgets import (
    Widget, Container, Card, Button, Progress, Text, RadioTable, RadioTabs, InputNumber, Grid, GridPlot, Table, Tabs,
    ProjectThumbnail, ClassesTable, TrainValSplits, Select, Input, Field, Editor, TabsDynamic, BindedInputNumber, Augmentations
)
import src.sly_globals as g


class TrainDashboard:
    def __init__(
            self, 
            project_id: int, 
            model,
            plots_titles: list = [],
            pretrained_weights: dict[list] = None,
            hyperparameters_categories: list[str] = ['general', 'checkpoints', 'optimizer', 'intervals', 'scheduler'],
            extra_hyperparams: dict[list] = {},
            hyperparams_edit_mode: Literal['ui', 'raw', 'all'] = 'all',
            show_augmentations_ui: bool = True,
            augmentation_templates: list[dict] = [],
            loggers: list = [sly.logger],
        ):
        """
        Easy configuritible training dashboard for NN training
        
        Parameters
        ----------
        project_id : int
            Source project ID
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
        augmentation_templates : list[dict] = TEMPLATES
            age of the person
        loggers : list = [sly.logger]
            list of loggers, which support log() method

        Methods
        -------
        train():
            Run training for your model. 
        """
        
        self._project = g.api.project.get_info_by_id(project_id)
        self._meta = sly.ProjectMeta.from_json(g.api.project.get_meta(project_id))

        self._model = model
        self._pretrained_weights = pretrained_weights
        self._hyperparameters = {}
        self._hyperparameters_categories = hyperparameters_categories
        self._extra_hyperparams = extra_hyperparams
        self._hyperparams_edit_mode = hyperparams_edit_mode
        self._show_augmentations_ui = show_augmentations_ui
        self._augmentation_templates = augmentation_templates
        self._loggers = loggers
        
        self._content = []
        self._project_preview = ProjectThumbnail(self._project)
        self._progress_bar_download_data = Progress(hide_on_finish=False)
        self._progress_bar_download_data.hide()
        self._text_download_data = Text('Project has been successfully downloaded', status='success')
        self._text_download_data.hide()
        self._button_download_dataset = Button('Download')
        @self._button_download_dataset.click
        def download_data():
            try:
                if sly.fs.dir_exists(g.project_dir):
                    pass
                else:
                    self._button_download_dataset.hide()
                    self._progress_bar_download_data.show()
                    sly.fs.mkdir(g.project_dir)
                    with self._progress_bar_download_data(message=f"Processing images...", total=g.project.items_count) as pbar:
                        sly.Project.download(
                            api=g.api, 
                            project_id=g.project_id, 
                            dest_dir=g.project_dir,
                            log_progress=True,
                            progress_cb=pbar.update,
                            only_image_tags=False, 
                            save_image_info=True)
                    self._progress_bar_download_data.hide()
                    self._text_download_data.show()
                g.project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
            except Exception as e:
                self._progress_bar_download_data.hide()
                self._button_download_dataset.show()
                self._text_download_data.set('Project download failed', status='error')
                self._text_download_data.show()
        self._project_card = Card(
            title="1. Input project", 
            description="This project will be used for training",
            content=Container([self._project_preview, self._progress_bar_download_data, self._text_download_data, self._button_download_dataset])
        )

        self._classes_table = ClassesTable(project_id=self._project.id)
        self._classes_table_card = Card(
            title="2. Classes table",
            description="Select classes, that should be used for training. Training supports only classes of shape Rectangle, other shapes are transformed to it automatically.",
            content=Container([self._classes_table]),
        )
        self._splits = TrainValSplits(project_id=self._project.id)
        self._unlabeled_images_selector = Select([
            Select.Item(value="keep", label="keep unlabeled images"),
            Select.Item(value="skip", label="ignore unlabeled images"),
        ])
        self._train_test_splits_card = Card(
            title="3. Train / Validation splits",
            description="Define how to split your data to train/val subsets",
            content=Container([
                self._splits, 
                Field(
                    title="What to do with unlabeled images", 
                    description="Sometimes unlabeled images may be used to reduce noise in predictions, sometimes it is a mistake in training data", 
                    content=self._unlabeled_images_selector),
                ]),
        )

        self._run_training_button = Button('Start training')
        @self._run_training_button.click
        def run_training():
            g.task_id = g.api.task.create_task_detached(g.workspace.id, task_type='training')

        self._progress_bar = Progress(message='Progress of training', hide_on_finish=False)
        self._logs_editor = Editor(
            'Training logs will be here...', 
            language_mode='plain_text', 
            restore_default_button=False, 
            readonly=True, 
            highlight_active_line=False,
            show_line_numbers=False)
        self._grid_plot = GridPlot(plots_titles, columns=3)
        self._logs_card = Card(title='Logs', content=self._logs_editor, collapsable=True)
        self._grid_plot_card = Card(title='Metrics', content=self._grid_plot, collapsable=True)
        self._training_card = Card(
            title="7. Training progress",
            description="Task progress, detailed logs, metrics charts, and other visualizations",
            content=Container([self._run_training_button, self._progress_bar, self._logs_card, self._grid_plot_card]),
        )
        
        self._content += [
            self._project_card,
            self._classes_table_card,
            self._train_test_splits_card,
            self.model_settings_card(),
            self.hyperparameters_card()
        ]
        if self._show_augmentations_ui:
            augmentations = Augmentations(templates=self._augmentation_templates)
            augmentations_card = Card(
                title="6. Training augmentations",
                description="Choose one of the prepared templates or provide custom pipeline",
                content=augmentations,
            )
            self._content.append(augmentations_card)
        self._content.append(self._training_card)

    def get_pretrained_weights_path(self):
        selected_trainig_mode = self._model_settings_tabs.get_active_tab()
        if selected_trainig_mode == 'Pretrained':
            path_col_index = self._weights_table.columns.index('Path')
            weights_path = self._weights_table.get_selected_row(StateJson())[path_col_index]
        elif selected_trainig_mode == 'Custom':
            weights_path = self._weights_path_input.get_value()
        else:
            weights_path = None
        return weights_path

    def model_settings_card(self):
        self._weights_path_input = Input(placeholder="Path to .pt file in Team Files")
        titles = ["Scratch", "Custom weights"]
        descriptions = ["Model training from scratch", "",]
        contents = [Text("Training from scratch", status="info"), self._weights_path_input]

        if self._pretrained_weights:
            self._weights_table = RadioTable(**self._pretrained_weights)
            titles.insert(1, "Pretrained")
            descriptions.insert(1, "Model pretrained checkpoints")
            contents.insert(1, self._weights_table)

        self._model_settings_tabs = RadioTabs(titles, contents, descriptions)
        return Card(
            title="4. Model settings",
            description="Choose model size or how weights should be initialized",
            content=Container([self._model_settings_tabs]),
        )

    def get_hyperparameters(self):
        hparams_from_file = self._hyperparameters_tab_dynamic.get_merged_yaml(as_dict=True)
        # converting OrderedDict to simple dict
        hparams_from_file = json.loads(json.dumps(hparams_from_file))
        hparams_from_ui = {key: widget.get_value() for (key, widget) in self._hyperparameters.items()}
        return hparams_from_file.update(hparams_from_ui)
    
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
            optimizers = {x:v for (x,v) in torch.optim.__dict__.items() if inspect.isclass(v)}
            hparams_widgets['optimizer'] = [
                dict(key='optimizer',
                    title='Optimizer', 
                    description='Setup corresponding learning rate for Adam in additional configuration, default values are provided for SGD', 
                    content=Select([Select.Item(value=key, label=key) for key in optimizers.keys()], size='small')),
                *self._extra_hyperparams.get('optimizer', [])
        ]
        if 'checkpoints' in self._hyperparameters_categories:
            hparams_widgets['checkpoints'] = [
                *self._extra_hyperparams.get('checkpoints', [])
            ]
        if 'intervals' in self._hyperparameters_categories:
            hparams_widgets['intervals'] = [
                dict(key='logging_interval',
                    title='Logging interval', 
                    description='How often metrics should be logged, increase if training data is small', 
                    content=InputNumber(1, min=1, max=10000, size='small')),
                dict(key='validation_interval',
                    title='Validation interval', 
                    description='How often to estimate the model on validation dataset', 
                    content=InputNumber(10, min=1, max=10000, size='small')),
                dict(key='сheckpoints_interval',
                    title='Checkpoints interval', 
                    description='How often to save the model weights', 
                    content=InputNumber(100, min=1, max=10000, size='small')),
                *self._extra_hyperparams.get('intervals', [])
            ]
        if 'scheduler' in self._hyperparameters_categories:
            schedulers = {x:v for (x,v) in torch.optim.lr_scheduler.__dict__.items() if inspect.isclass(v)}
            hparams_widgets['scheduler'] = [
                dict(key='scheduler',
                    title='Scheduler', 
                    description='Learning rate scheduler', 
                    content=Select([Select.Item(value=key, label=key) for key in schedulers.keys()], size='small')),
                *self._extra_hyperparams.get('scheduler', [])
            ]

        return hparams_widgets 

    def hyperparameters_card(self):
        card_content = []

        labels = []
        contents = []
        if self._hyperparams_edit_mode in ('ui', 'all'):
            for tab_label, widgets in self.hyperparameters_ui().items():
                grid_content = []
                if len(widgets) == 0:
                    continue
                for hparam in widgets:
                    grid_content.append(Field(hparam['content'], hparam['title'], hparam['description']))
                    self._hyperparameters[hparam['key']] = hparam['content']
                hyperparameters_grid = Grid(grid_content, columns=1)
                labels.append(tab_label.capitalize())
                contents.append(hyperparameters_grid)
            hparams_tabs = Tabs(labels, contents)
            card_content.append(hparams_tabs)
        if self._hyperparams_edit_mode in ('raw', 'all'):
            self._hyperparameters_file_selector = Select([
                    Select.Item(value='/Users/ruslantau/Desktop/example.yml', label="Scratch mode | Recommended hyperparameters for training from scratch"),
                    Select.Item(value="/Users/ruslantau/Desktop/example2.yml", label="Finetune mode | Recommended hyperparameters for model finutuning"),
                ])
            @self._hyperparameters_file_selector.value_changed
            def hyperparams_file_changed(value):
                print(f"New file is: {value}")
                # TODO
                # self._hyperparameters_tab_dynamic.reinit(value)

            self._hyperparameters_tab_dynamic = TabsDynamic(self._hyperparameters_file_selector.get_value())
            hyperparameters_selector_field = Field(title="Hyperparameters file", 
                                description="Choose from provided files or select own from team files", 
                                content=self._hyperparameters_file_selector)
            card_content += [hyperparameters_selector_field, self._hyperparameters_tab_dynamic]

        return Card(
            title="5. Traning hyperparameters",
            description="Define general settings and advanced configuration (learning rate, augmentations, ...)",
            content=Container(card_content)
        )

    def get_optimizer(self, name: str):
        if name is None:
            return None
        return torch.optim.__dict__[name]

    def get_scheduler(self, name: str):
        return torch.optim.lr_scheduler.__dict__[name]

    def get_transforms(self):
        if self.show_augmentations_ui:
            # TODO get transforms from augmentation UI component
            return None
        else:
            return ToTensor()

    def train(self):
        raise NotImplementedError('You need to define train loop for your model')

    def inference():
        pass
    
    def log(self, value_to_log):
        for logger in self._loggers():
            logger.log(value_to_log)

    def run(self):
        return sly.Application(
            layout=Container(
                widgets=self._content,
                direction="vertical", gap=20)
        )