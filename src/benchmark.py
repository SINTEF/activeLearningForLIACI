from os import mkdir, path
from shutil import rmtree
from subprocess import run
import config as cnf
from pipeline import pipeline_start
from datetime import datetime 

def path_create():
    bench_start = datetime.now().strftime('%Y_%m_%d_%H%M')
    dir_path = cnf.bench_dir + bench_start + '/'
    if path.isdir(dir_path):
        rmtree(dir_path) 
    mkdir(dir_path)
    return dir_path 

def main():
    seed = 0
    dir_path  = path_create()

    epochs =        [ 
                        25,
                        30,
                        35 
                    ]

    batch_size =    [ 
                        50, 
                        100,
                        200,
                    ]
    
    lr =            [ 
                        4e-4, 
                        2e-4, 
                        2e-5,
                    ]

    v_split =       [ 
                        0.1, 
                        0.15, 
                        0.20 
                    ]
    version2 =      [ 
                        False, 
                        True 
                    ]
    
    for e in epochs:
        for b in batch_size:
            for l in lr:
                for vs in v_split:
                    for v in version2:
                        
                        d_path = dir_path + f'epochs-{e}_batch_size-{b}_lr-{l}_v_split-{vs}_v2-{v}/' 
                        mkdir(d_path)

                        pipeline_start(
                            epochs=e, lr=l, v_split=vs, batch_size=b, 
                            seed=seed,
                            version_2=v, 
                            path=d_path, 
                            save_option='y',
                        )   

    


    
    
    pass

if __name__ == '__main__':
    main()