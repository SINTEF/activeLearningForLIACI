# activeLearningForLIACI

## Requirements
* Python 3.10 
* pipenv (if not installed, run `python3 -m pip install pipenv` to install pipenv through pip)


All of the actions below MUST be executed in the `src/` folder

## Installing the correct python packages and running the enviornment
1. Run `pipenv install` to install all the required python packages
2. Run `pipenv shell` to enter the virtual environment shell


## Downloading the dataset
1. Run `python3 download_dataset.py`

## Starting the web interface and running inference on a video
1. Run `python3 app.py` to start the server 
2. Go to http://127.0.0.1:8050 in your browser to view the website
3. Click the upload button on the web page and select your video to upload
4. Wait for the server to process the video
5. View the video and see which labels are classified live in the video
