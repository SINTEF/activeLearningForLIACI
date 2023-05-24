from wget import download
from os import mkdir, path, remove
from shutil import rmtree
from zipfile import ZipFile
from prints import printo, printw

def main():

    
    if path.isdir('../LIACi_dataset_pretty/'):
        printw('Dataset already downloaded')
    else:
        url = 'https://liaci.sintef.cloud/download_data/data.zip'
        filename = download(url, out='../')
        with ZipFile(filename, 'r') as zf:
            zf.extractall('../') # Open and extract zip file
            print('ZIP downloaded')

    image_thumbs = '../LIACi_dataset_pretty/image_thumbs/'
    if path.isdir(image_thumbs):
        rmtree(image_thumbs)
        print('Removed ' + image_thumbs)
    
    masks = '../LIACi_dataset_pretty/masks/'
    if path.isdir(masks):
        rmtree(masks)
        print('Removed ' + masks)

    data = '../data.zip'
    if path.isfile(data):
        remove(data)
        print('Removed ' + data)

    printo('Dataset ready')
        
    

    


if __name__ == '__main__':
    main()