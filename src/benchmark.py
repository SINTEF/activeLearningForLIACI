from os import path, mkdir
import config as cnf
from datetime import datetime 

def path_create():
    bench_start = datetime.now().strftime('%Y_%m_%d_%H%M')
    path = cnf.bench_dir + bench_start
    mkdir(path)
    return path 

epochs = 30

