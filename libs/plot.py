import inspect
import json
import os.path
from datetime import datetime

class Plot():
    def __init__(self):
        self.project = 'fl'
        self.name = 'test-run'
        self.time = ''
        self.file = ''
        
    def log(self, _json):
        with open(self.file,'r+') as file:
            file_data = json.load(file)
            for key, value in _json.items():
                file_data[self.name][self.time][key] = value
            file.seek(0)
            json.dump(file_data, file, indent = 4)
            
    def alog(self, parent, _json):
        with open(self.file,'r+') as file:
            file_data = json.load(file)
            if parent not in file_data[self.name][self.time]:
                file_data[self.name][self.time][parent] = {}
            for key, value in _json.items():
                file_data[self.name][self.time][parent][key] = value
            file.seek(0)
            json.dump(file_data, file, indent = 4)

def init(name = None, project = None):
    plot = Plot()
    if name is not None:
        plot.name = name
    if project is not None:
        plot.project = project

    plotdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../out/plots/' + plot.project
    plot.file = plotdir + '/' + plot.name + ".json"
    plot.time = datetime.today().isoformat()

    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

    if not os.path.isfile(plot.file):
        with open(plot.file, 'w') as f:
            json.dump({plot.name: {plot.time: {}}}, f, indent = 4)
    else:
        add_plot(plot.file, plot.name, plot.time)
        
    return plot

def add_plot(file, name, time):
    with open(file,'r+') as file:
        file_data = json.load(file)
        file_data[name][time] = {}
        file.seek(0)
        json.dump(file_data, file, indent = 4)