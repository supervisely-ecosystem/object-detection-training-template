import os
from dotenv import load_dotenv
from typing import Optional, List
from supervisely.app.content import DataJson, StateJson
import json

import supervisely as sly
from supervisely.app.widgets import Widget
from supervisely.app.widgets import (
    Container, Card, Button, Progress, Text, RadioTable, RadioTabs, InputNumber, Grid, GridPlot, Table,
    ProjectThumbnail, ClassesTable, TrainValSplits, Select, Input, Field, Editor, TabsDynamic, BindedInputNumber
    )
import torch 
from torch.utils.data import DataLoader

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

class TrainDashboard:
    def __init__(
            self, 
            project_id: int, 
            model = None,
            enable_augmentations_ui: bool = True,
            extra_hyperparams: list = [],
            show_hyperparams_text_editor: bool = True,
        ):
        self._project = api.project.get_info_by_id(project_id)
        self._meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        self._model = model
        self._hyperparameters = {}
        self._extra_hyperparams = extra_hyperparams
        self._show_hyperparams_text_editor = show_hyperparams_text_editor
        self._content = []

        self._project_preview = ProjectThumbnail(self._project)
        self._project_card = Card(
            title="1. Input project", 
            description="This project will be used for training",
            content=self._project_preview
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
        self._progress_bar = Progress(message='Progress of training', hide_on_finish=False)
        self._logs_editor = Editor(
            'Training logs will be here...', 
            language_mode='plain_text', 
            restore_default_button=False, 
            readonly=True, 
            highlight_active_line=False,
            show_line_numbers=False)
        self._grid_plot = GridPlot(['GIoU', 'Objectness', 'Classification', 'Pr + Rec', 'mAP'], columns=3)
        self._logs_card = Card(title='Logs', content=self._logs_editor, collapsable=True)
        self._grid_plot_card = Card(title='Metrics', content=self._grid_plot, collapsable=True)
        self._training_card = Card(
            title="6. Training progress",
            description="Task progress, detailed logs, metrics charts, and other visualizations",
            content=Container([self._run_training_button, self._progress_bar, self._logs_card, self._grid_plot_card]),
        )
        

        button = Button('test')
        @button.click
        def run_metrics_generation():
            p = self.get_pretrained_weights_path()

        self._content += [
            self._project_card,
            self._classes_table_card,
            self._train_test_splits_card,
            self.model_settings_card(),
            self.hyperparameters_card(),
            self._training_card,
            button
        ]

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

    def pretrained_weights(self):
        return {
            'columns': ['Name', 'Description', 'Path'],
            'rows': [
                ['Unet', 'Vanilla Unet', '/mnt/weights/unet.pth'],
                ['Unet-11', 'VGG16', '/mnt/weights/unet11.pth'], 
                ['Unet-16', 'VGG11', '/mnt/weights/unet16.pth']
            ]
        }

    def model_settings_card(self):
        self._weights_path_input = Input(placeholder="Path to .pt file in Team Files")
        self._weights_table = RadioTable(**self.pretrained_weights())
        self._model_settings_tabs = RadioTabs(
            titles=["Scratch", "Pretrained", "Custom weights"],
            contents=[
                Text("Training from scratch", status="info"),
                self._weights_table,
                self._weights_path_input
            ],
            descriptions=[
                "Model training from scratch",
                "Model pretrained checkpoints",
                "",
            ],
        )
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
        return [
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
                 content=InputNumber(8, min=1, size='small')),
            dict(key='logging_frequency',
                 title='Logging frequency', 
                 description='How often metrics should be logged, increase if training data is small', 
                 content=InputNumber(1, min=1, max=10000, size='small')),
            dict(key='optimizer',
                 title='Optimizer', 
                 description='Setup corresponding learning rate for Adam in additional configuration, default values are provided for SGD', 
                 content=Select([
                    Select.Item(value="sgd", label="SGD"),
                    Select.Item(value="adam", label="ADAM"),
                    ], size='small')),
            *self._extra_hyperparams
        ]
        
    def hyperparameters_card(self):
        grid_content = []
        card_content = []

        for hparam in self.hyperparameters_ui():
            grid_content.append(Field(hparam['content'], hparam['title'], hparam['description']))
            self._hyperparameters[hparam['key']] = hparam['content']
        hyperparameters_grid = Grid(grid_content, columns=1)

        if self._show_hyperparams_text_editor:
            self._hyperparameters_file_selector = Select([
                    Select.Item(value='/Users/ruslantau/Desktop/example.yml', label="Scratch mode | Recommended hyperparameters for training from scratch"),
                    Select.Item(value="/Users/ruslantau/Desktop/example2.yml", label="Finetune mode | Recommended hyperparameters for model finutuning"),
                ])
            self._hyperparameters_tab_dynamic = TabsDynamic(self._hyperparameters_file_selector.get_value())
            hyperparameters_selector_field = Field(title="Hyperparameters file", 
                                description="Choose from provided files or select own from team files", 
                                content=self._hyperparameters_file_selector)
            card_content += [hyperparameters_selector_field, self._hyperparameters_tab_dynamic]
        return Card(
            title="5. Traning hyperparameters",
            description="Define general settings and advanced configuration (learning rate, augmentations, ...)",
            content=Container([hyperparameters_grid, *card_content]),
        )

    def get_optimizer(self):
        # TODO default params fix
        if self._hyperparameters['optimizer'] == 'Adadelta':
            return torch.optim.Adadelta(
                self._model.parameters(), 
                lr=self._hyperparameters.get('lr', 1.0), 
                rho=self._hyperparameters.get('rho', 0.9), 
                eps=self._hyperparameters.get('eps', 1e-06), 
                weight_decay=self._hyperparameters('weight_decays', 0), 
                foreach=self._hyperparameters.get('foreach', None), 
                maximize=self._hyperparameters.get('maximize', False),
            )
        elif self._hyperparameters['optimizer'] == 'Adagrad':
            return torch.optim.Adagrad(
                self._model.parameters(), 
                lr=self._hyperparameters.get('lr', 0.01), 
                lr_decay=self._hyperparameters.get('lr_decay', 0), 
                weight_decay=self._hyperparameters('weight_decays', 0), 
                foreach=self._hyperparameters.get('foreach', None), 
                initial_accumulator_value=self._hyperparameters.get('initial_accumulator_value', 0), 
                eps=self._hyperparameters.get('eps', 1e-10), 
                maximize=self._hyperparameters.get('maximize', False),
            )
        elif self._hyperparameters['optimizer'] == 'Adam':
            return torch.optim.Adam(
                self._model.parameters(), 
                lr=self._hyperparameters.get('lr', 0.001), 
                betas=self._hyperparameters.get('betas', (0.9, 0.999)), 
                eps=self._hyperparameters.get('eps', 1e-08), 
                weight_decay=self._hyperparameters('weight_decays', 0), 
                amsgrad=self._hyperparameters('weight_decays', False), 
                foreach=self._hyperparameters.get('foreach', None), 
                maximize=self._hyperparameters.get('maximize', False),
                capturable=self._hyperparameters.get('capturable', False), 
                differentiable=self._hyperparameters.get('differentiable', False), 
                fused=self._hyperparameters.get('fused', False),
            )
        elif self._hyperparameters['optimizer'] == 'AdamW':
            return torch.optim.AdamW(
                self._model.parameters(), 
                lr=self._hyperparameters.get('lr', 0.001), 
                betas=self._hyperparameters.get('betas', (0.9, 0.999)), 
                eps=self._hyperparameters.get('eps', 1e-08), 
                weight_decay=self._hyperparameters('weight_decays', 0.01), 
                amsgrad=self._hyperparameters('weight_decays', False), 
                maximize=self._hyperparameters.get('maximize', False),
                foreach=self._hyperparameters.get('foreach', None), 
                capturable=self._hyperparameters.get('capturable', False), 
            )
        elif self._hyperparameters['optimizer'] == 'SparseAdam':
            return torch.optim.SparseAdam(
                self._model.parameters(), 
                ...
            )
        elif self._hyperparameters['optimizer'] == 'Adamax':
            return torch.optim.Adamax(
                self._model.parameters(), 
                ...
            )
        elif self._hyperparameters['optimizer'] == 'ASGD':
            return torch.optim.ASGD(
                self._model.parameters(), 
                ...
            )
        elif self._hyperparameters['optimizer'] == 'LBFGS':
            return torch.optim.LBFGS(
                self._model.parameters(), 
                ...
            )
        elif self._hyperparameters['optimizer'] == 'NAdam':
            return torch.optim.NAdam(
                self._model.parameters(), 
                ...
            )
        elif self._hyperparameters['optimizer'] == 'RAdam':
            return torch.optim.RAdam(
                self._model.parameters(), 
                ...
            )
        elif self._hyperparameters['optimizer'] == 'RMSprop':
            return torch.optim.RMSprop(
                self._model.parameters(), 
                ...
            )
        elif self._hyperparameters['optimizer'] == 'Rprop':
            return torch.optim.Rprop(
                self._model.parameters(), 
                ...
            )
        elif self._hyperparameters['optimizer'] == 'SGD':
            return torch.optim.SGD(
                self._model.parameters(), 
                lr=self._hyperparameters['lr'], 
                momentum=self._hyperparameters.get('momentum', 0), 
                dampening=self._hyperparameters.get('dampening', 0), 
                weight_decay=self._hyperparameters('weight_decays', 0), 
                nesterov=self._hyperparameters('nesterov', False), 
                maximize=self._hyperparameters.get('maximize', False),
                foreach=self._hyperparameters.get('foreach', None), 
                differentiable=self._hyperparameters.get('differentiable', False),
            )
        elif self._hyperparameters['optimizer'] == 'RAdam':
            raise NotImplementedError(f"Selected optimizer {self._hyperparameters['optimizer']} not inplemented")

    def train(self):
        classes = self._classes_table.get_selected_classes()
        train_set, val_set = self._splits.get_splits()
        hparams = self.get_hyperparameters()
        optimizer = self.get_optimizer()
        device = f"cuda:{hparams['device']}" if hparams['device'].isdigit() else hparams['device']

        train_loader = DataLoader(train_set, batch_size=hparams['batch_size'])
        val_loader = DataLoader(val_set, batch_size=hparams['batch_size'])

        pretrained_weights_path = self.get_pretrained_weights_path()
        if pretrained_weights_path:
            model = torch.load_state_dict(pretrained_weights_path)
        else:
            model = Model()

        model.train()
        
        for epoch in range(self._hyperparameters['number_of_epochs']):
            train_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
            
            self._grid_plot.add_scalar('Loss/train', train_loss, epoch)
            # self._progress_bar.
            # self._progress_bar.
    def inference():
        pass

    def run(self):
        return sly.Application(
            layout=Container(
                widgets=dashboard._content,
                direction="vertical", gap=20)
        )
        
    





project_id = sly.env.project_id()
dashboard = TrainDashboard(project_id=project_id)
app = dashboard.run()