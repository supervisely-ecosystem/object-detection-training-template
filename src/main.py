import json

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import supervisely as sly
from supervisely.app.widgets import InputNumber
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.sly_globals as g
from dashboard import TrainDashboard

def get_pseudo_random(x, increasing=False): 
    if increasing:
        return np.random.normal(loc=0.5, scale=1.0) + np.sqrt(x)
    else:
        return np.random.normal(loc=0.5, scale=1.0) - np.sqrt(x)


class CustomDataset(Dataset):
    def __init__(self, items_infos, transforms=None):
        self.items_infos = items_infos
        self.transforms = transforms
    
    def __getitem__(self, index):
        image = sly.image.read(self.items_infos[index].img_path)
        with open(self.items_infos[index].ann_path, 'r') as f:
            ann = sly.Annotation.from_json(json.loads(f.read()), g.project_meta)
        target = dict(
            label=ann.labels,
            bbox = ann.labels
        )

        if self.transforms:
            res_meta, image, res_ann = sly.imgaug_utils.apply(self.transforms, g.project_meta, image, ann)

        target = {}
        return (image, target)

    def __len__(self):
        return len(self.items_infos)


class CustomTrainDashboard(TrainDashboard):
    def train(self):
        classes = self._classes_table.get_selected_classes()
        train_set, val_set = self.get_splits()
        hparams = self.get_hyperparameters()
        optimizer = self.get_optimizer(name=hparams['optimizer']['name'])
        optimizer = optimizer(
            self._model.parameters(),
            hparams['optimizer']['lr'],
            hparams['optimizer']['foreach'],
            hparams['optimizer']['maximize'],
            hparams['optimizer']['eps'],
        )
        scheduler = self.get_scheduler(name=hparams['scheduler']['name'])
        # scheduler = scheduler(**hparams['scheduler'])
        device = f"cuda:{hparams['general']['device']}" if hparams['general']['device'].isdigit() else hparams['general']['device']
        
        train_dataset = CustomDataset(train_set, transforms=self.get_transforms())
        val_dataset = CustomDataset(val_set)
        train_loader = DataLoader(train_dataset, batch_size=hparams['general']['batch_size'], num_workers=hparams['general']['workers_number'])
        val_loader = DataLoader(val_dataset, batch_size=hparams['general']['batch_size'], num_workers=hparams['general']['workers_number'])

        pretrained_weights_path = self.get_pretrained_weights_path()
        if pretrained_weights_path:
            self._model = torch.load_state_dict(pretrained_weights_path)

        with self._progress_bar(message=f"Training...", total=hparams['general']['number_of_epochs']) as pbar:
            self._model.train()
            for epoch in range(hparams['general']['number_of_epochs']):
                train_loss = 0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    # inputs, targets = inputs.to(device), targets.to(device)
                    # optimizer.zero_grad()
                    # outputs = self._model(inputs)
                    # loss = torch.nn.functional.mse_loss(outputs, targets)
                    # loss.backward()
                    # optimizer.step()

                    # train_loss += loss.item()
                    # _, train_predicted = outputs.max(1)
                    
                    train_loss = get_pseudo_random(epoch, increasing=False)
                    train_metric = get_pseudo_random(epoch, increasing=False)
                    pass
                
                model.eval()
                if hparams['intervals'].get('validation', False):
                    if epoch % hparams['intervals']['validation'] == 0:
                        for batch_idx, (inputs, targets) in enumerate(val_loader):
                            with torch.no_grad():
                                # outputs = self._model(inputs)
                                # loss = torch.nn.functional.mse_loss(outputs, targets)
                                # val_loss += loss.item()
                                # _, val_predicted = outputs.max(1)
                                val_loss = get_pseudo_random(epoch, increasing=True)
                                val_metric = get_pseudo_random(epoch, increasing=True)
                                pass

                if hparams['intervals'].get('сheckpoints_interval', False):
                    if epoch % hparams['intervals']['сheckpoints'] == 0:
                        torch.save(self._model.state_dict(), self._root_dir / f'model_epoch_{epoch}.pth')

                if epoch % hparams['intervals'].get('logging_interval', 1) == 0:
                    self.log('add_scalar', tag='Loss/train', scalar_value=train_loss, global_step=epoch)
                    self.log('add_scalar', tag='Loss/val', scalar_value=val_loss, global_step=epoch)
                    self.log('add_scalar', tag='IoU/train', scalar_value=train_metric, global_step=epoch)
                    self.log('add_scalar', tag='IoU/val', scalar_value=val_metric, global_step=epoch)
                
                if scheduler:
                    # scheduler.step()
                    pass
                
                self.log('add_text', tag='Main logs', text_string=f"Epoch: {epoch}")
                pbar.update(1)
            pbar.set_description_str("Training has been successfully finished")


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
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
my_logger = SummaryWriter('./runs')

dashboard = CustomTrainDashboard(
    model=model, 
    hyperparams_edit_mode='all',
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
    pretrained_weights=PRETRAINED_WEIGHTS,
    augmentation_templates=AUG_TEMPLATES,
    plots_titles=GRID_PLOT_TITLES,
    show_augmentations_ui=True,
    loggers=[my_logger]
)
app = dashboard.run()