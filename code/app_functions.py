from dash import html
import datetime
import cv2
# from opencv import 

class VideoStream:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del_(self):
        self.video.release()
    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def parse_upload(content, filename, date):
    # print(len(content.split(',')) if content else None)
    if content:
        print("found countetn")
        cap = cv2.VideoCapture(content)
        TNF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(TNF)
    return html.Div([
        html.Video(src=content if content else None,
            controls=True,
            style={
                'width':'50%'
            }),
        html.H5('Filename' if not filename else filename),
        html.H6('Date' if not date else datetime.datetime.fromtimestamp(date)),
        # html.Hr(),
        # html.Div('raw Content'),
    ])