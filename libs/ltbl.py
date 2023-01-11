import inspect
import os.path
from datetime import datetime
import pandas as pd

class LogTbl():
    def __init__(self):
        self.project = 'fl'
        self.name = 'test-run'
        self.file = ''
        self.data_dict = ''
        
    def ar(self, data_dict):
        for key, value in self.data_dict.items():
            if key in data_dict:
                self.data_dict[key].append(data_dict[key])
            else:
                self.data_dict[key].append("NA")
                
        df = pd.DataFrame.from_dict(self.data_dict)
        df.to_csv(self.file)

def init(data_dict, name = None, project = None):
    tbl = LogTbl()
    tbl.data_dict = data_dict
    if name is not None:
        tbl.name = name
    if project is not None:
        tbl.project = project

    csvdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../out/csvs/' + tbl.project
    tbl.file = csvdir + '/' + tbl.name + "-" + datetime.today().isoformat()+ ".csv"

    if not os.path.exists(csvdir):
        os.makedirs(csvdir)

    return tbl