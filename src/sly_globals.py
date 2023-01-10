import os
import sys
from pathlib import Path

import supervisely as sly
from dotenv import load_dotenv
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

DEBUG = False

root_source_dir = str(Path(sys.argv[0]).parents[5])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)
source_path = os.path.join(root_source_dir, 'src')
sly.logger.info(f"App source directory: {source_path}")
sys.path.append(source_path)

api = sly.Api()
project_id = sly.env.project_id()
project = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
workspace = api.workspace.get_info_by_id(project.workspace_id)
team = api.team.get_info_by_id(workspace.team_id)
project_fs: sly.Project = None
task_id = None

project_dir = os.path.join(root_source_dir, "sly_project")
artifacts_dir = os.path.join(root_source_dir, 'artifacts')
checkpoints_dir = os.path.join(root_source_dir, 'checkpoints')
