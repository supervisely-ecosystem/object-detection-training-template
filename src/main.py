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
        Select.Item(value="/Users/ruslantau/Desktop/example.yml", label="Scratch mode | Recommended hyperparameters for training from scratch"),
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
logs_editor = Editor('Training logs will be here...', language_mode='plain_text')
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


# @button.click
# def calculate_stats():
#     with progress(message=f"Processing images...", total=project.items_count) as pbar:
#         for dataset in api.dataset.get_list(project.id):
#             images = api.image.get_list(dataset.id)
#             for batch in sly.batched(images):
#                 batch_ids = [image.id for image in batch]
#                 annotations = api.annotation.download_json_batch(dataset.id, batch_ids)
#                 for image, ann_json in zip(batch, annotations):
#                     ann = sly.Annotation.from_json(ann_json, meta)
#                     stats.increment(dataset, image, ann)
#                     pbar.update(1)
#     lines = []
#     for class_name, x, y in stats.get_series():
#         lines.append({"name": class_name, "x": x, "y": y})
#     chart.add_series_batch(lines)
#     button.hide()
#     chart.show()


# @chart.click
# def refresh_images_table(datapoint: HeatmapChart.ClickedDataPoint):
#     table.loading = True
#     labeled_image.clean_up()
#     df = stats.get_table_data(cls_name=datapoint.series_name, obj_count=datapoint.x)
#     table.read_pandas(df)
#     click_info.description = f"Images with {datapoint.x} object(s) of class {datapoint.series_name}"
#     table.loading = False


# @table.click
# def show_image(datapoint: Table.ClickedDataPoint):
#     if datapoint.button_name == None:
#         return
#     labeled_image.loading = True
#     image_id = datapoint.row["id"]
#     image = api.image.get_info_by_id(image_id)
#     ann_json = api.annotation.download_json(image_id)
#     ann = sly.Annotation.from_json(ann_json, meta)
#     labeled_image.set(title=image.name, image_url=image.preview_url, ann=ann, image_id=image_id)
#     labeled_image.loading = False
