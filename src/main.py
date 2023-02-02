import os
import json
import random 

import torch
import torch.nn as nn
import numpy as np
import imgaug.augmenters as iaa
import torch.nn.functional as F
import supervisely as sly
from supervisely.app.widgets import InputNumber, Augmentations
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

import src.sly_globals as g
from src.dashboard import TrainDashboard


class CustomTrainDashboard(TrainDashboard):
    def train(self):
        classes = self._classes_table.get_selected_classes()
        train_set, val_set = self.get_splits()
        hparams = self.get_hyperparameters()
        optimizer = self.get_optimizer(name=hparams['optimizer']['name'])
        optimizer = optimizer(
            self.model.parameters(),
            hparams['optimizer']['lr'],
            hparams['optimizer']['foreach'],
            hparams['optimizer']['maximize'],
            hparams['optimizer']['eps'],
        )
        device = f"cuda:{hparams['general']['device']}" if hparams['general']['device'].isdigit() else hparams['general']['device']
        # extra hparam to scale loss
        C = hparams['general']['C']

        transforms = self.get_transforms()
        train_dataset = CustomDataset(train_set, transforms=transforms, classes=classes, image_size=hparams['general']['input_image_size'])
        val_dataset = CustomDataset(val_set, classes=classes, image_size=hparams['general']['input_image_size'])
        train_loader = DataLoader(train_dataset, batch_size=hparams['general']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hparams['general']['batch_size'])

        pretrained_weights_path = self.get_pretrained_weights_path()
        if pretrained_weights_path:
            self.model = torch.load_state_dict(pretrained_weights_path)
        
        with self.progress_bar(message=f"Training...", total=hparams['general']['number_of_epochs']) as pbar:
            self.model.train()
            for epoch in range(hparams['general']['number_of_epochs']):
                train_total_samples = 0
                train_sum_loss = 0
                train_correct = 0 
                for batch_idx, (images, classes, bboxes) in enumerate(train_loader):
                    batch_size = images.shape[0]
                    images = images.to(device).float()
                    classes = classes.to(device)
                    bboxes = bboxes.to(device).float()

                    pred_classes, pred_bboxes = self.model(images)
                    loss_class = F.cross_entropy(pred_classes, classes, reduction="sum")
                    loss_bb = F.l1_loss(pred_bboxes, bboxes, reduction="none").sum(1)
                    loss_bb = loss_bb.sum()
                    loss = loss_class + loss_bb / C
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_total_samples += batch_size
                    train_sum_loss += loss.item()
                    
                    _, pred = torch.max(pred_classes, 1)
                    train_correct += pred.eq(classes).sum().item()
                train_loss = train_sum_loss / train_total_samples
                train_accuracy = train_correct / train_total_samples

                if hparams['intervals'].get('validation', False):
                    if epoch % hparams['intervals']['validation'] == 0:
                        model.eval()
                        val_total_samples = 0
                        val_sum_loss = 0
                        val_correct = 0 
                        for batch_idx, (images, classes, bboxes) in enumerate(val_loader):
                            batch_size = images.shape[0]
                            images = images.to(device).float()
                            classes = classes.to(device)
                            bboxes = bboxes.to(device).float()

                            pred_classes, pred_bboxes = self.model(images)
                            loss_class = F.cross_entropy(pred_classes, classes, reduction="sum")
                            loss_bb = F.l1_loss(pred_bboxes, bboxes, reduction="none").sum(1)
                            loss_bb = loss_bb.sum()
                            loss = loss_class + loss_bb / C

                            val_sum_loss += loss.item()
                            val_total_samples += batch_size

                            _, pred = torch.max(pred_classes, 1)
                            val_correct += pred.eq(classes).sum().item()
                        val_loss = val_sum_loss / val_total_samples
                        val_accuracy = val_correct / val_total_samples

                if hparams['intervals'].get('сheckpoints_interval', False):
                    if epoch % hparams['intervals']['сheckpoints'] == 0:
                        torch.save(self.model.state_dict(), os.path.join(g.checkpoints_dir, f'model_epoch_{epoch}.pth'))

                if epoch % hparams['intervals'].get('logging_interval', 1) == 0:
                    self.log('add_scalar', tag='Loss/train', scalar_value=train_loss, global_step=epoch)
                    self.log('add_scalar', tag='Loss/val', scalar_value=val_loss, global_step=epoch)
                    self.log('add_scalar', tag='Accuracy/train', scalar_value=train_accuracy, global_step=epoch)
                    self.log('add_scalar', tag='Accuracy/val', scalar_value=val_accuracy, global_step=epoch)
                
                
                self.log('add_text', tag='Main logs', text_string=f"Epoch: {epoch}\t|\tTrain loss: {train_loss:.3f}\t|\tVal loss: {val_loss:.3f}\t|\tTrain accuracy: {train_accuracy:.3f}\t|\tVal accuracy: {val_accuracy:.3f}")
                pbar.update(1)
            pbar.set_description_str("Training has been successfully finished")


class CustomDataset(Dataset):
    def __init__(self, items_infos, classes, image_size, transforms=None):
        self.items_infos = items_infos
        self.classes = classes
        self.image_size = image_size
        self.transforms = transforms
    
    def __getitem__(self, index):
        image = sly.image.read(self.items_infos[index].img_path)
        meta = g.project_meta
        with open(self.items_infos[index].ann_path, 'r') as f:
            ann = sly.Annotation.from_json(json.loads(f.read()), meta)
        ann, meta = Augmentations.convert_ann_to_bboxes(ann, meta)

        if self.transforms:
            res_meta, image, ann = sly.imgaug_utils.apply(self.transforms, meta, image, ann)

        meta, image, ann = sly.imgaug_utils.apply(
            iaa.Sequential([
                iaa.Resize({"height": self.image_size[1], "width": self.image_size[0]})
            ]), 
            meta, 
            image, 
            ann
        )

        label = random.choice(ann.labels)
        class_id = self.classes.index(label.obj_class.name)
        bbox = label.geometry.to_bbox()
        bbox = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
    
        image = np.rollaxis(image, 2)
        return image, class_id, bbox

    def __len__(self):
        return len(self.items_infos)


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        resnet = models.resnet34(weights=models.resnet.ResNet34_Weights.IMAGENET1K_V1)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)


HPARAMS_TEMPLATES = [
    {'label': 'Scratch mode | Recommended hyperparameters for training from scratch', 'value':'/Users/ruslantau/Desktop/example.yml'},
    {'label': 'Finetune mode | Recommended hyperparameters for model finutuning', 'value':'/Users/ruslantau/Desktop/example2.yml'},
]
AUG_TEMPLATES = [
    {'label': 'Light', 'value':'aug_templates/light.json'},
    {'label': 'Light + corrupt', 'value':'aug_templates/light_corrupt.json'},
    {'label': 'Medium', 'value':'aug_templates/medium.json'},
    {'label': 'Medium + corrupt', 'value':'aug_templates/medium_corrupt.json'},
    {'label': 'Hard', 'value':'aug_templates/hard.json'},
    {'label': 'Hard + corrupt', 'value':'aug_templates/hard_corrupt.json'},
]
PRETRAINED_WEIGHTS = {
    'columns': ['Name', 'Description', 'Path'],
    'rows': [
        ['Unet', 'Vanilla Unet', '/mnt/weights/unet.pth'],
        ['Unet-11', 'VGG16', '/mnt/weights/unet11.pth'], 
        ['Unet-16', 'VGG11', '/mnt/weights/unet16.pth']
    ]
}

model = CustomModel()
my_logger = SummaryWriter(g.tensorboard_runs_dir)

dashboard = CustomTrainDashboard(
    model=model, 
    hyperparams_edit_mode='ui',
    extra_hyperparams={
        'general': [
            dict(key='C',
                title='Bbox loss scale', 
                description='Divide bbox_loss for this value', 
                content=InputNumber(1000, min=1, max=100000, size='small')),
        ],
    },
    # pretrained_weights=PRETRAINED_WEIGHTS,
    augmentation_templates=AUG_TEMPLATES,
    plots_titles=['Loss', 'Accuracy'],
    show_augmentations_ui=True,
    task_type='detection',
    loggers=[my_logger]
)
app = dashboard.run()