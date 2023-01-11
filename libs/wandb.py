import inspect
import json
import os.path
import wandb

class WB():
    def __init__(self):
        self.project = 'fl'
        self.name = 'test-run'
        self.config = {}
        
    def log(self, _json):
        wandb.log(_json)
        
def init(name = None, project = None, config = None):
    wb = WB()
    if project is not None:
        wb.project = project
    if name is not None:
        wb.name = name
    if config is not None:
        wb.config = config

    wandb.init(project=wb.project, 
               dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../out',
               name=wb.name,
               config=wb.config)
    
    return wb