import torch
import supervisely as sly
from supervisely.app.widgets import InputNumber
from torch.utils.data import Dataset, DataLoader

import src.sly_globals as g
from dashboard import TrainDashboard


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


class CustomTrainDashboard(TrainDashboard):
    def train(self):
        classes = self._classes_table.get_selected_classes()
        train_set, val_set = self._splits.get_splits()
        hparams = self.get_hyperparameters()
        optimizer = self.get_optimizer(name=hparams['optimizer']['name'])
        # optimizer = optimizer(**hparams['optimizer'])
        scheduler = self.get_scheduler(name=hparams['scheduler']['name'])
        # scheduler = scheduler(**hparams['scheduler'])
        device = f"cuda:{hparams['general']['device']}" if hparams['general']['device'].isdigit() else hparams['general']['device']
        
        train_dataset = CustomDataset(train_set['images'], train_set['labels'], transform=self.get_transforms())
        val_dataset = CustomDataset(val_set['images'], val_set['labels'], transform=self.get_transforms())
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], num_workers=hparams.get('workers_number', 0))
        val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], num_workers=hparams.get('workers_number', 0))

        pretrained_weights_path = self.get_pretrained_weights_path()
        if pretrained_weights_path:
            self._model = torch.load_state_dict(pretrained_weights_path)

        self._model.train()
        
        with self._progress_bar(message=f"Processing items...", total=hparams['general']['number_of_epochs']) as pbar:
            for epoch in range(hparams['general']['number_of_epochs']):
                train_loss = 0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = torch.nn.functional.mse_loss(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, train_predicted = outputs.max(1)
                
                if hparams.get('validation_interval', False):
                    if epoch % hparams['validation_interval'] == 0:
                        for batch_idx, (inputs, targets) in enumerate(val_loader):
                            self._model.eval()
                            outputs = self._model(inputs)
                            loss = torch.nn.functional.mse_loss(outputs, targets)
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

                self.log(f"Epoch: {epoch}")
                pbar.update(1)
            pbar.set_description_str("Training has been successfully finished")


class CustomModel():
    def forward():
        pass

    
HPARAMS_TEMPLATES = [
    {'label': 'Scratch mode | Recommended hyperparameters for training from scratch', 'value':'/Users/ruslantau/Desktop/example.yml'},
    {'label': 'Finetune mode | Recommended hyperparameters for model finutuning', 'value':'/Users/ruslantau/Desktop/example2.yml'},
]
AUG_TEMPLATES = [
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


model = CustomModel()

dashboard = CustomTrainDashboard(
    model=model, 
    hyperparams_edit_mode='ui',
    extra_hyperparams={
        'general': [
            dict(key='additional hparam',
                title='Additional general hparam ', 
                description='Description for additional hparam', 
                content=InputNumber(10, min=1, max=100000, size='small')),
        ],
        'scheduler': [
            dict(key='additional hparam',
                title='Additional scheduler hparam ', 
                description='Description for additional hparam', 
                content=InputNumber(10, min=1, max=100000, size='small')),
        ],
    },
    hyperparams_templates=HPARAMS_TEMPLATES,
    pretrained_weights=PRETRAINED_WEIGHTS,
    augmentation_templates=AUG_TEMPLATES,
    plots_titles=GRID_PLOT_TITLES,
    show_augmentations_ui=True,
    loggers=[],
)
app = dashboard.run()