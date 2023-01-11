# hdfl
HDC-based FL

* If Conda enviroment
    * conda env create --name hdfl --file=cenv.yml
        * it will also add this conda env in your base jupyter notebook, look for [reference](https://towardsdatascience.com/how-to-set-up-anaconda-and-jupyter-notebook-the-right-way-de3b7623ea4a)
    * conda activate hdfl
    * open host_ip:port for opening jupyter notebook
        * jupyter notebook --no-browser --ip="*" --port=xxxx --NotebookApp.token='xx' --NotebookApp.iopub_data_rate_limit=1.0e1000
        
* Run src/hdfl/fl-irbr.ipynb or src/hdfl/fl-irbr.py and make changes as required by looking paramaters from cfgs/fedargs.py
