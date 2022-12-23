import os
from dotenv import load_dotenv

import supervisely as sly
from supervisely.app.widgets import (
    Container, Card, Button, Progress, Text, Tabs, RadioTabs, InputNumber, Grid, GridPlot,
    ProjectThumbnail, ClassesTable, TrainValSplits, Select, Input, Field, Editor, TabsDynamic
    )
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

project_id = sly.env.project_id()
project = api.project.get_info_by_id(project_id)
meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

project_preview = ProjectThumbnail(project)
input_card = Card(
    title="1. Input project", 
    description="This project will be used for training",
    content=project_preview
)

table = ClassesTable(project_id=project_id)
table_card = Card(
    title="2. Classes table",
    description="Select classes, that should be used for training. Training supports only classes of shape Rectangle, other shapes are transformed to it automatically.",
    content=Container([table]),
)

splits = TrainValSplits(project_id=project_id)
select_items = Select([
    Select.Item(value="keep", label="keep unlabeled images"),
    Select.Item(value="skip", label="ignore unlabeled images"),
])
train_test_splits_card = Card(
    title="3. Train / Validation splits",
    description="Define how to split your data to train/val subsets",
    content=Container([
        splits, 
        Field(
            title="What to do with unlabeled images", 
            description="Sometimes unlabeled images may be used to reduce noise in predictions, sometimes it is a mistake in training data", 
            content=select_items),
        ]),
)

weights_path_input = Input(placeholder="Path to .pt file in Team Files")
model_settings_tabs = RadioTabs(
    titles=["Pretrained", "Custom weights"],
    contents=[
        Text("Pretrained", status="info"),
        weights_path_input
    ],
    descriptions=[
        "Model pretrained checkpoints",
        "",
    ],
)
model_settings_card = Card(
    title="4. Model settings",
    description="Choose model size or how weights should be initialized",
    content=Container([model_settings_tabs]),
)

yaml_data = """
#YAML
# Hyperparameters for COCO training from scratch
# python train.py --batch 40 --cfg yolov5m.yaml --weights '' --data coco.yaml --img 640 --epochs 300
# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4

warmup:
  warmup_epochs: 3.0  # warmup epochs (fractions ok)
  warmup_momentum: 0.8  # warmup initial momentum
  warmup_bias_lr: 0.1  # warmup initial bias lr

losses:
  box: 0.05  # box loss gain
  cls: 0.5  # cls loss gain
  cls_pw: 1.0  # cls BCELoss positive_weight
  obj: 1.0  # obj loss gain (scale with pixels)
  obj_pw: 1.0  # obj BCELoss positive_weight
  # anchors: 3  # anchors per output layer (0 to ignore)
  fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)

thresholds:
  iou_t: 0.20  # IoU training threshold
  anchor_t: 4.0  # anchor-multiple threshold

image:
  hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
  hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
  hsv_v: 0.4  # image HSV-Value augmentation (fraction)
  degrees: 0.0  # image rotation (+/- deg)
  translate: 0.1  # image translation (+/- fraction)
  scale: 0.5  # image scale (+/- gain)
  shear: 0.0  # image shear (+/- deg)
  perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
  flipud: 0.0  # image flip up-down (probability)
  fliplr: 0.5  # image flip left-right (probability)
  mosaic: 1.0  # image mosaic (probability)
  mixup: 0.0  # image mixup (probability)

# extra augs
augmentations:
  image:
    channels: 
      spectrum_1: 5
      spectrum_2: 3
      spectrum_3: 1
      spectrum_4: 2
      spectrum_5: 0
      spectrum_6: 4
      spectrum_7: 6

boolean: Yes
string: "25"
infinity: .inf
neginf: -.Inf 
not: .NAN 
null: ~
"""
hyperparameters_container = Grid([
    Field(
        title="Number of epochs", 
        description="Total count epochs for training", 
        content=InputNumber(10, min=1, max=100000, size='small')),
    Field(
        title="Batch size", 
        description="total batch size for all GPUs. Use the largest batch size your GPU allows. For example: 16 / 24 / 40 / 64 (batch sizes shown for 16 GB devices)", 
        content=InputNumber(8, min=6, max=100000, size='small')),
    Field(
        title="Input image size (in pixels)", 
        description="Image is resized to square", 
        content=InputNumber(512, min=64, size='small')),
    Field(
        title="Device", 
        description="Cuda device, i.e. 0 or 0,1,2,3 or cpu, or keep empty to select automatically", 
        content=Input('0', size='small')),
    Field(
        title="Number of workers", 
        description="Maximum number of dataloader workers, use 0 for debug", 
        content=InputNumber(8, min=1, size='small')),
    Field(
        title="Logging frequency", 
        description="How often metrics should be logged, increase if training data is small", 
        content=InputNumber(1, min=1, max=10000, size='small')),
    Field(
        title="Optimizer", 
        description="Setup corresponding learning rate for Adam in additional configuration, default values are provided for SGD", 
        content=Select([
            Select.Item(value="sgd", label="SGD"),
            Select.Item(value="adam", label="ADAM"),
        ], size='small'))
    ], 
    columns=3)
hyperparameters_file_selector = Select([
        Select.Item(value=yaml_data, label="Scratch mode | Recommended hyperparameters for training from scratch"),
        Select.Item(value="/Users/ruslantau/Desktop/example2.yml", label="Finetune mode | Recommended hyperparameters for model finutuning"),
    ])
hyperparameters_tab_dynamic = TabsDynamic(hyperparameters_file_selector.get_value())
hyperparameters_card = Card(
    title="5. Traning hyperparameters",
    description="Define general settings and advanced configuration (learning rate, augmentations, ...)",
    content=Container([
        hyperparameters_container, 
        Field(
            title="Hyperparameters file", 
            description="Choose from provided files or select own from team files", 
            content=hyperparameters_file_selector),
        hyperparameters_tab_dynamic
        ]),
)

run_training_button = Button('Start training')
progress_bar = Progress(message='Progress of training', hide_on_finish=False)
logs_editor = Editor(
    'Training logs will be here...', 
    language_mode='plain_text', 
    restore_default_button=False, 
    readonly=True, 
    highlight_active_line=False,
    show_line_numbers=False)
grid_plot = GridPlot(['GIoU', 'Objectness', 'Classification', 'Pr + Rec', 'mAP'], columns=3)
logs_card = Card(title='Logs', content=logs_editor, collapsable=True)
grid_plot_card = Card(title='Metrics', content=grid_plot, collapsable=True)
training_card = Card(
    title="6. Training progress",
    description="Task progress, detailed logs, metrics charts, and other visualizations",
    content=Container([run_training_button, progress_bar, logs_card, grid_plot_card]),
)

app = sly.Application(
    layout=Container(
        widgets=[
            input_card, 
            table_card, 
            train_test_splits_card, 
            model_settings_card, 
            hyperparameters_card,
            training_card
            ], 
        direction="vertical", gap=20)
)