from flask import Flask, Response, render_template
import cv2
from helpers.helpers import *
import time

app = Flask(__name__)
cap = cv2.VideoCapture(0)

labels, model, colors, output_layers = load_model()

@app.route('/')
def index():
    return render_template("index.html")

def gen(video):
    font = cv2.FONT_HERSHEY_PLAIN
    ret, frame = video.read()
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (220, 220), (0, 0, 0), True,
                                 crop=False)  # reduce size to improve speed
    model.setInput(blob)
    outs = model.forward(output_layers)

    boxes, confidences, class_ids = draw_boxes(outs, height, width)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # removes noisy boxes

    while True:
        ret, frame = video.read()
        if round(time.time()) % 2 == 0:
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (220, 220), (0, 0, 0), True,
                                         crop=False)  # reduce size to improve speed
            model.setInput(blob)
            outs = model.forward(output_layers)

            boxes, confidences, class_ids = draw_boxes(outs, height, width)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # removes noisy boxes

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(labels[class_ids[i]])
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)


        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route("/video")
def video():
    global cap
    return Response(gen(cap), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)

