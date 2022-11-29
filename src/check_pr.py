from os import walk
from pipeline import pipeline_start
from data import load_from_coco
from prints import printc

def get_dict_from_file(path):
    values = {}
    with open(path, 'r') as f:
        for line in f:
            if not '=' in line:
                continue
            (k,v) = line.replace(' ', '').replace('\n','').split('=')
            values[k] = v
    return values

def main():
    
    bench_dir = '../benchmarks/' + '2022_11_29_1306' + '/'
    dirs = walk(bench_dir)
    # subdirs = [x[0] for x in dirs]
    subdirs = [ x[0].split('/')[-1] for x in dirs if len(x[0].split('/')) == 4 and x[0].split('/')[-1]]

    X, Y = load_from_coco()
    seed=0

    for d in subdirs:
        if not d:
            continue

        params_path = bench_dir+ d + '/model/params.txt'
        params = get_dict_from_file(params_path)
        print(params)
        printc(f'Processing: {bench_dir + d}')
        pipeline_start(
            model_path=bench_dir + d + '/model/', 
            path=bench_dir + d + '/',
            transfer_learning=False,
            evaluate=True,
            save_option='n',
            X=X,
            Y=Y,
            seed=seed, # int(params['seed']), # use this if new bench is ran
            v_split=float(params['v_split']),
            lr=float(params['lr']),
            epochs=int(params['epochs']),
            version_2=params['v2']==True,
        )


if __name__ == '__main__':
    main()