import os
import sys
import time
from pathlib import Path

import supervisely as sly
from dotenv import load_dotenv
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

root_source_dir = str(Path(sys.argv[0]).parents[5])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

api = sly.Api()
project_id = sly.env.project_id()
project_fs: sly.Project = None
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

project = api.project.get_info_by_id(project_id)
workspace = api.workspace.get_info_by_id(project.workspace_id)
# team.id will be used for storaging app results in team files
team = api.team.get_info_by_id(workspace.team_id)

project_dir = os.path.join(root_source_dir, "sly_project")
data_dir = os.path.join(root_source_dir, "data")
checkpoints_dir = os.path.join(data_dir, 'checkpoints')
tensorboard_runs_dir = os.path.join(data_dir, 'tensorboard_runs')
remote_data_dir = f"/train_dashboard/{project.name}/runs/{time.strftime('%Y-%m-%d %H:%M:%S')}"
os.environ["SLY_APP_DATA_DIR"] = data_dir
