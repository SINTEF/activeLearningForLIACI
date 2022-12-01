from os import walk, mkdir, path
from pipeline import pipeline_start
from data import load_from_coco
from prints import printc

def get_dict_from_file(f_path):
    values = {}
    with open(f_path, 'r') as f:
        for line in f:
            if not '=' in line:
                continue
            (k,v) = line.replace(' ', '').replace('\n','').split('=')
            values[k] = v
    return values

def main():
    
    bench_dir = '../benchmarks/' + '2022_11_29_1306' + '/'
    dirs = walk(bench_dir)
    subdirs = [ x[0].split('/')[-1] for x in dirs if len(x[0].split('/')) == 4 and x[0].split('/')[-1]]

    X, Y = load_from_coco()
    seed=0

    for d in subdirs:
        if not d:
            continue
        
        dir_path = bench_dir + d + '/'
        params_path = dir_path + 'model/params.txt'
        pdf_path = dir_path + 'pdfs/'

        if not path.isdir(pdf_path):
            mkdir(pdf_path)
            
        params = get_dict_from_file(params_path)

        printc(f'Processing: {dir_path}')
        pipeline_start(
            model_path=dir_path + 'model/', 
            path=dir_path,
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