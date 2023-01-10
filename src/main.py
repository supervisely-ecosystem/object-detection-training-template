import supervisely as sly
from supervisely.app.widgets import (
    Container, Card, Button, Progress, Text, Tabs, RadioTabs, InputNumber, Grid, GridPlot,
    ProjectThumbnail, ClassesTable, TrainValSplits, Select, Input, Field, Editor, TabsDynamic
    )
from torch.utils.data import Dataset, DataLoader

import src.sly_globals as g
from dashboard import TrainDashboard


TEMPLATES = [
    {'label': 'Hard + corrupt', 'value':'aug_templates/hard_corrupt.json'},
    {'label': 'Hard', 'value':'aug_templates/hard.json'},
    {'label': 'Light + corrupt', 'value':'aug_templates/light_corrupt.json'},
    {'label': 'Light', 'value':'aug_templates/light.json'},
    {'label': 'Medium + corrupt', 'value':'aug_templates/medium_corrupt.json'},
    {'label': 'Medium', 'value':'aug_templates/medium.json'},
]
PRETRAINED_WEIGHTS = {
    'columns': ['Name', 'Description', 'Path'],
    'rows': [
        ['Unet', 'Vanilla Unet', '/mnt/weights/unet.pth'],
        ['Unet-11', 'VGG16', '/mnt/weights/unet11.pth'], 
        ['Unet-16', 'VGG11', '/mnt/weights/unet16.pth']
    ]
}
GRID_PLOT_TITLES = ['Loss', 'IoU']


class CustomDataset(Dataset):
    def __init__(self, images, classes, transform=None):
        self.images = images
        self.targets = classes
        self.transform = transform

    def __getitem__(self, item):
        image = self.images[item]
        target = self.targets[item]

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.images)


project_id = sly.env.project_id()
# my_model = Model()

class CustomTrainDashboard(TrainDashboard):
    def train(self):
        classes = self._classes_table.get_selected_classes()
        train_set, val_set = self._splits.get_splits()
        hparams = self.get_hyperparameters()
        optimizer = self.get_optimizer(name=hparams['optimizer'])(**hparams['optimizer_params'])
        scheduler = self.get_scheduler(name=hparams.get('optimizer', None))
        device = f"cuda:{hparams['device']}" if hparams['device'].isdigit() else hparams['device']
        
        train_dataset = CustomDataset(train_set['images'], train_set['labels'], transform=self.get_transforms())
        val_dataset = CustomDataset(val_set['images'], val_set['labels'], transform=self.get_transforms())
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], num_workers=hparams.get('workers_number', 0))
        val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], num_workers=hparams.get('workers_number', 0))

        pretrained_weights_path = self.get_pretrained_weights_path()
        if pretrained_weights_path:
            self._model = torch.load_state_dict(pretrained_weights_path)

        self._model.train()
        
        with self._progress_bar(message=f"Processing items...", total=20) as pbar:
            for epoch in range(hparams['number_of_epochs']):
                train_loss = 0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, train_predicted = outputs.max(1)
                
                if hparams.get('validation_interval', False):
                    if epoch % hparams['validation_interval'] == 0:
                        for batch_idx, (inputs, targets) in enumerate(val_loader):
                            self._model.eval()
                            outputs = self._model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                            _, val_predicted = outputs.max(1)
                            pass

                if hparams.get('сheckpoints_interval', False):
                    if epoch % hparams['сheckpoints_interval'] == 0:
                        torch.save(self._model.state_dict(), self._root_dir / f'model_epoch_{epoch}.pth')

                if epoch % hparams.get('logging_interval', 1) == 0:
                    self._grid_plot.add_scalar('Loss/train', train_loss, epoch)
                
                if scheduler:
                    scheduler.step()

                # TODO
                # self._progress_bar
                self.log(f"Epoch: {epoch}")
                pbar.update(1)
            pbar.set_description_str("Training has been successfully finished")


dashboard = CustomTrainDashboard(
    project_id=project_id, 
    model=None, 
    hyperparams_edit_mode='ui',
    extra_hyperparams={
        'general': [
            dict(key='additional hparam',
                title='Additional general hparam ', 
                description='Description for additional hparam', 
                content=InputNumber(10, min=1, max=100000, size='small')),
        ],
        'intervals': [
            dict(key='additional hparam',
                title='Additional intervals hparam ', 
                description='Description for additional hparam', 
                content=InputNumber(10, min=1, max=100000, size='small')),
        ],
    },
    pretrained_weights=PRETRAINED_WEIGHTS,
    augmentation_templates=TEMPLATES,
    plots_titles=GRID_PLOT_TITLES,
    show_augmentations_ui=False,
    loggers=[],
)
app = dashboard.run()