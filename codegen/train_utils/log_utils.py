import warnings
import wandb

class WandbLogger:
    def __init__(self, project_name="CourseProjectTrial", run_name=None):
        wandb.login(relogin=True, key="f0de171d1703a95c6d727f2918c794da293d02c7")
        self.project_name = project_name
        self.run = wandb.init(project=self.project_name, name=run_name)

    def log(self, data, step):
        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()

class DummyLogger():
    def __init__(self, *args, **kwargs):
        pass
    def log(self, data, step):
        pass
    def finish(self):
        pass

class PrintLogger:
    def __init__(self):
        pass

    def log(self, data, step):
        print(data)

    def finish(self):
        pass


def load_logger(report_to):
    name2logger = {
        None: DummyLogger,
        "none": DummyLogger,
        "wandb": WandbLogger,
        "print": PrintLogger,
    }

    assert report_to in name2logger, "Logger {report_to} not found"
    return name2logger[report_to]()