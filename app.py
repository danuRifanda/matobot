from flask import Flask, render_template, Response, url_for
import cv2
import numpy as np

app = Flask(__name__)
camera = None

def detect_colors2(frame):
    # convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([0, 128, 161])
    upper_red = np.array([15, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([0, 128, 161])
    upper_red = np.array([15, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red, upper_red)

    # define range of green color in HSV
    lower_green = np.array([21, 134, 106])
    upper_green = np.array([40, 255, 248])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # combine masks for red and green colors
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask, mask_green)

    # bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # get contour area
        area = cv2.contourArea(contour)

        # filter contours by area
        if area > 1000:
            # draw rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if cv2.countNonZero(mask_red1[y:y+h, x:x+w]) > 0:
                # add text "merah" to the frame
                cv2.putText(frame, "Matang", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            elif cv2.countNonZero(mask_green[y:y+h, x:x+w]) > 0:
                cv2.putText(frame, "Mentah", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame

def detect_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range_merah=np.array([0, 147, 94], np.uint8)
    upper_range_merah=np.array([179, 255, 255], np.uint8)
    lower_range_hijau=np.array([30, 73, 134], np.uint8)
    upper_range_hijau=np.array([52, 184, 205], np.uint8)
    mask_red = cv2.inRange(hsv,lower_range_merah,upper_range_merah)
    mask_green = cv2.inRange(hsv,lower_range_hijau,upper_range_hijau)

    if cv2.countNonZero(mask_red) > cv2.countNonZero(mask_green):
        _,mask1=cv2.threshold(mask_red,254,255,cv2.THRESH_BINARY)
        cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            x=1000
            if cv2.contourArea(c)>x:
                x1,y1,w1,h1=cv2.boundingRect(c)
                cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
                cv2.putText(frame, "Matang", (x1, y1+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
                cv2.putText(frame,("DETECT"),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    else:
        _,mask1=cv2.threshold(mask_green,254,255,cv2.THRESH_BINARY)
        cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            x=600
            if cv2.contourArea(c)>x:
                x,y,w,h=cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frame, "Belum Matang", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
                cv2.putText(frame,("DETECT"),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    
    return frame

def generate_frames():
    while True:
        if camera is not None:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def gen():
    while True:
        if camera is not None:
            success, frame = camera.read()
            if not success:
                break
            else:
                # detect color in frame
                frame = detect_colors2(frame)

                # encode frame as jpeg
                ret, jpeg = cv2.imencode('.jpg', frame)

                # convert jpeg to bytes and yield as response
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    global camera
    cv2.destroyAllWindows()
    camera = None
    return render_template('index-8(1).html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera')
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        return 'Camera started'
    else:
        return 'Camera already started'

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        cv2.destroyAllWindows()
        camera = None
        return 'Camera stopped'
    else:
        return 'Camera already stopped'

@app.route('/new_page')
def new_page():
    return render_template('new_page.html')

@app.route('/deteksi')
def deteksi():
    return render_template('deteksi.html')

if __name__ == '__main__':
    app.run(debug=True)
