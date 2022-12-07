from wget import download
from os import mkdir, path
from shutil import rmtree
from zipfile import ZipFile
from prints import printo, printw

def main():

    url = 'https://liaci.sintef.cloud/download_data/data.zip'
    if path.isdir('../LIACi_dataset_pretty/'):
        printw('Dataset already downloaded. Exiting...')
        exit()

    filename = download(url, out='../')
    
    with ZipFile(filename, 'r') as zf:
        zf.extractall('../') # Open and extract zip file

    image_thumbs = '../LIACi_dataset_pretty/image_thumbs/'
    if path.isdir(image_thumbs):
        rmtree(image_thumbs)
        print('Removed ' + image_thumbs)
    
    masks = '../LIACi_dataset_pretty/masks/'
    if path.isdir(masks):
        rmtree(masks)
        print('Removed ' + masks)

    printo('Dataset downloaded')
        
    

    


if __name__ == '__main__':
    main()