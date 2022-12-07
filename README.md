# activeLearningForLIACI

## Requirements
* Python 3.10 <= 
* pipenv (if not installed, run `python3 -m pip install pipenv` to install pipenv through pip)

## Starting the web interface 
All of the actions below have to be executed in the `src/` folder for this section (skip this section if server is already running)
1. Run `pipenv install` to install all the requirements
2. Run `pipenv shell` to enter the virtual environment shell
3. Run `python3 app.py` to start the server 


## Running inference on a video
0. Go to http://127.0.0.1:8050 in your browser to view the website
1. Click the upload button on the web page and select your video to upload
2. Wait for the server to process the video
3. View the video and see which labels are classified live in the video