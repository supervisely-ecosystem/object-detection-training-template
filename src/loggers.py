
from tensorboard.compat.proto.event_pb2 import SessionLog
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto import event_pb2
from torch.utils.tensorboard import FileWriter as OriginalFileWriter
from torch.utils.tensorboard import SummaryWriter as OriginalSummaryWriter


class FileWriter(OriginalFileWriter):
    def add_summary(self, summary, global_step=None, walltime=None):
        """Adds a `Summary` protocol buffer to the event file.
        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.

        Args:
          summary: A `Summary` protocol buffer.
          global_step: Number. Optional global step value for training process
            to record with the summary.
          walltime: float. Optional walltime to override the default (current)
            walltime (from time.time()) seconds after epoch
        """
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)

        # Supervisely 
        if event in ('add_scalar') and global_step is not None:
            self.grid_plot.add_scalar(event.tag, event.data, global_step)
            # self.grid_plot.add_scalar('Loss/train', train_loss, global_step)


class TensorboardSummaryWriter(OriginalSummaryWriter):
    def _get_file_writer(self):
        """Returns the default FileWriter instance. Recreates it if closed."""
        if self.all_writers is None or self.file_writer is None:
            self.file_writer = FileWriter(
                self.log_dir, self.max_queue, self.flush_secs, self.filename_suffix
            )
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            if self.purge_step is not None:
                most_recent_step = self.purge_step
                self.file_writer.add_event(
                    Event(step=most_recent_step, file_version="brain.Event:2")
                )
                self.file_writer.add_event(
                    Event(
                        step=most_recent_step,
                        session_log=SessionLog(status=SessionLog.START),
                    )
                )
                self.purge_step = None
        return self.file_writer


# from functools import wraps

# def callback(function):
#     @wraps(function)
#     def wrapper(*args, **kwargs):
#         print('Arguments:', args, kwargs) 
#         return function(*args, **kwargs)  
#     return wrapper
# 
# TensorboardSummaryWriter.add_audio = callback(TensorboardSummaryWriter.add_audio)