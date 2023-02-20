import os
import json
import random 
from typing import Literal, List, Dict, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import imgaug.augmenters as iaa
import torch.nn.functional as F
import supervisely as sly
from supervisely.app.widgets import InputNumber
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

import src.sly_globals as g
from src.dashboard import ObjectDetectionTrainDashboad


class CustomTrainDashboard(ObjectDetectionTrainDashboad):
    def train(self):
        # getting selected classes from UI
        classes = self._classes_table.get_selected_classes()
        # getting configuration of splits from UI
        train_set, val_set = self.get_splits()
        # getting hyperparameters from UI
        hparams = self.get_hyperparameters()
        # initialized with selected hyperparameters pytorch optimizer will be returned 
        optimizer = self.get_optimizer(hparams)
        device = hparams.general.device
        # extra hparam to scale loss
        C = hparams.general.C

        # getting selected augmentation from UI
        transforms = self.get_transforms()
        train_dataset = CustomDataset(
            train_set, 
            project_meta=g.project_fs.meta, 
            classes=classes, 
            input_size=hparams.general.input_image_size,
            transforms=transforms, 
        )
        val_dataset = CustomDataset(
            val_set, 
            project_meta=g.project_fs.meta, 
            classes=classes, 
            input_size=hparams.general.input_image_size
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=hparams.general.batch_size, 
            shuffle=True,
            num_workers=hparams.general.workers_number
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=hparams.general.batch_size,
            num_workers=hparams.general.workers_number
        )

        # it will return None if pretrained model weights isn't selected in UI
        if self.pretrained_weights_path:
            self.model.load_state_dict(torch.load(self.pretrained_weights_path))
            sly.logger.info('Model checkpoint successfully loaded.')
        
        model.to(device)
        with self.progress_bar(message=f"Training...", total=hparams.general.number_of_epochs) as pbar:
            self.model.train()
            # change training and eval loops for your model if needed
            for epoch in range(hparams.general.number_of_epochs):
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
                
                if hasattr(hparams.intervals, 'validation'):
                    if epoch % hparams.intervals.validation == 0:
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

                if hasattr(hparams.intervals, 'сheckpoints'):
                    if epoch % hparams.intervals.сheckpoints == 0:
                        temp_checkpoint_path = os.path.join(g.checkpoints_dir, f'model_epoch_{epoch}.pth')
                        torch.save(self.model.state_dict(), temp_checkpoint_path)
                        self.log('add_text', tag='Main logs', text_string=f"Model saved at:\n{temp_checkpoint_path}")

                if epoch % hparams.intervals.logging == 0:
                    # common method to logging your values to dashboard
                    # you log values only by your own logger if it setted just call it by class name
                    # self.loggers.YOUR_LOGGER.add_scalar(tag='Loss/train', scalar_value=train_loss, global_step=epoch)
                    self.log('add_scalar', tag='Loss/train', scalar_value=train_loss, global_step=epoch)
                    self.log('add_scalar', tag='Loss/val', scalar_value=val_loss, global_step=epoch)
                    self.log('add_scalar', tag='Accuracy/train', scalar_value=train_accuracy, global_step=epoch)
                    self.log('add_scalar', tag='Accuracy/val', scalar_value=val_accuracy, global_step=epoch)
                
                self.log('add_text', tag='Main logs', text_string=f"Epoch: {epoch}\t|\tTrain loss: {train_loss:.3f}\t|\tVal loss: {val_loss:.3f}\t|\tTrain accuracy: {train_accuracy:.3f}\t|\tVal accuracy: {val_accuracy:.3f}")
                pbar.update(1)
            pbar.set_description_str("Training has been successfully finished")

class CustomDataset(Dataset):
    def __init__(self, 
                 items: sly.ImageInfo, 
                 project_meta: sly.ProjectMeta, 
                 classes: List[str], 
                 input_size: Union[List, Tuple], 
                 transforms: Optional[iaa.Sequential] = None):
        self.items = items
        self.project_meta = project_meta
        self.classes = classes
        self.input_size = input_size
        self.transforms = transforms
    
    def __getitem__(self, index: int):
        image = sly.image.read(self.items[index].img_path)
        with open(self.items[index].ann_path, 'r') as f:
            ann = sly.Annotation.from_json(json.loads(f.read()), self.project_meta)

        if self.transforms:
            res_meta, image, ann = sly.imgaug_utils.apply(self.transforms, self.project_meta, image, ann)

        meta, image, ann = sly.imgaug_utils.apply(
            iaa.Sequential([
                iaa.Resize({"height": self.input_size[1], "width": self.input_size[0]})
            ]), 
            self.project_meta, 
            image, 
            ann
        )

        # our simple model can work only with 1 bbox per image
        # you need to change code if your model more complicated
        label = random.choice(ann.labels)
        class_id = self.classes.index(label.obj_class.name)
        bbox = label.geometry.to_bbox()
        bbox = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
    
        image = np.rollaxis(image, 2)
        return image, class_id, bbox

    def __len__(self):
        return len(self.items)


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        resnet = models.resnet34()
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


model = CustomModel()
# my_logger = SummaryWriter(g.tensorboard_runs_dir)

dashboard = CustomTrainDashboard(
    model=model, 
    plots_titles=['Loss', 'Accuracy'],
    extra_hyperparams={
        'general': [
            dict(key='C',
                title='Bbox loss scale', 
                description='Divide bbox_loss for this value', 
                content=InputNumber(1000, min=1, max=100000, size='small')),
        ],
    },
    # loggers=[my_logger]
)
app = dashboard.run()