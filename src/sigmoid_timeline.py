from os import walk, mkdir, path
import cv2
from model import hullifier_load
from tqdm import tqdm
import numpy as np
from data import get_cat_lab
import matplotlib.pyplot as plt
from itertools import product



def create_timeline(model, path, video_path):
    
    vid = cv2.VideoCapture(video_path)
    labels = get_cat_lab()

    tnf = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames
    fps = vid.get(cv2.CAP_PROP_FPS)
    duration = tnf / fps

    succ, image = vid.read()
    if not succ:
        raise Exception("Can't parse video")

    # Predict on frames
    frames = []
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print('Fetching images from video...')
    for i in tqdm(range(tnf)):
        succ, im = vid.read()
        if not succ:
            raise Exception("Can't parse video image")
        im = cv2.resize(im, (224,224))
        frames.append(im)

    frames = np.array(frames)
    predictions = model.predict(frames)

    x_ax = np.arange(tnf)
    figaro, axs  = plt.subplots(3,3)
    
    xy = product(np.arange(3), np.arange(3))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for l, label, (x, y), color in zip(predictions.T, labels, xy, colors):
        axs[x][y].plot(x_ax, l, c=color)
        axs[x][y].set_xlabel('Frame number')
        axs[x][y].set_ylabel(label)
        # axs[x][y].legend()
        axs[x][y].set_ylim([0,1])
    
    
    figaro.tight_layout()
    figaro.savefig(path + 'timeline.png')
    figaro.savefig(path + 'pdfs/timeline.pdf')

    return figaro
        
def main():
    video_path = '../videoplayback.mp4'
    bench_dir = '../benchmarks/' + '2022_11_29_1306' + '/'
    dirs = walk(bench_dir)
    subdirs = [ x[0].split('/')[-1] for x in dirs if len(x[0].split('/')) == 4 and x[0].split('/')[-1]]
    
    dir_path = bench_dir + subdirs[0] + '/'
    model_path =  dir_path + 'model/'
    model = hullifier_load(model_path)
    create_timeline(model, dir_path, video_path)
    


        
        
if __name__ == '__main__':
    main()