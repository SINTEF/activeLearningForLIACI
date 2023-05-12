from os import mkdir, path
from shutil import rmtree
from subprocess import run
import utils.config as cnf
from pipeline import pipeline_start
from datetime import datetime 

import tensorflow as tf

from data import load_from_coco


def path_create():
    bench_start = datetime.now().strftime('%Y_%m_%d_%H%M')
    dir_path = cnf.bench_dir + bench_start + '/'
    if path.isdir(dir_path):
        rmtree(dir_path) 
    mkdir(dir_path)
    return dir_path 

def main():
    tf.get_logger().setLevel('ERROR')

    seed = 0

    epochs =        [ 
                        # 25,
                        30,
                        # 35 
                    ]

    batch_size =    [ 
                        50, 
                        # 100,
                        # 200,
                    ]
    
    lr =            [ 
                        4e-4, 
                        # 5e-4, 
                        # 2e-5,
                    ]

    v_split =       [ 
                        0.1, 
                        # 0.2
                    ]
    version2 =      [ 
                        False, 
                        # True 
                    ]
    

    X, Y = load_from_coco()
    dir_path  = path_create()

    for e in epochs:
        for b in batch_size:
            for l in lr:
                for vs in v_split:
                    for v in version2:
                        
                        d_path = dir_path + f'epochs-{e}_batchsize-{b}_lr-{l}_vsplit-{vs}_v2-{v}/' 
                        mkdir(d_path)

                        pipeline_start(
                            epochs=e, lr=l, v_split=vs, batch_size=b, 
                            seed=seed,
                            version_2=v, 
                            path=d_path, 
                            save_option='y',
                            X=X,
                            Y=Y,
                            evaluate=False,
                        )   

    
    print(f'finished writing benchmarks to {dir_path}')

    
    
    pass

if __name__ == '__main__':
    main()