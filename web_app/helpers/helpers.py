import numpy as np
import cv2


def load_model():
    with open("../labels.txt", "r") as f:
        labels = f.read().split("\n")

    config = "../yolov3.cfg"
    weights = "../yolov3.weights"
    net = cv2.dnn.readNet(weights, config)

    layer_names = net.getLayerNames()
    output_layers = []

    for i in net.getUnconnectedOutLayers():
        output_layers.append(layer_names[i - 1])

    colors = np.random.uniform(0, 255, size=(len(labels), 3))

    return labels, net, colors, output_layers


def draw_boxes(outs, height, width):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids
