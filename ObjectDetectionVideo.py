import cv2
import numpy as np

cap = cv2.VideoCapture("videos/Driving_Downtown_New_York_City.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter("output/output.mp4", fourcc, 10.0, (640, 360))


def load_yolo():
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    classes = []
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print(output_layers)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def display_blob(blob):
    '''
      Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def preprocessing_image(img):
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=(320, 320), swapRB=True, crop=False)
    return blob


def detect_objects(blob, net, outputLayers):
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return outputs



def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def non_maximum_supression(boxes, confs):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.5, nms_threshold=0.4)
    return indexes


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = non_maximum_supression(boxes=boxes, confs=confs)
    font = cv2.QT_FONT_NORMAL
    font_class_scale = 0.6
    font_conf_scale = 0.3
    objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            objects.append(label)
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(img, label, (x, y - 5), font, font_class_scale, color, 1)
            cv2.putText(img, str(round((confs[0]*100), 2))+"%", (x, y +h + 10), font, font_conf_scale, color, 1)
    unique_classes, amount_unique_classes = np.unique(objects, return_counts=True)
    y = 0
    for i, c in enumerate(unique_classes):
        y += 15
        cv2.putText(img, "{}:{} ".format(c, amount_unique_classes[i]), (0, y),
                    font, font_class_scale, colors[classes.index(c)], 1)


if __name__ == '__main__':
    model, classes, colors, output_layers = load_yolo()
    while True:
        crt, frame = cap.read()
        if crt:
            height, width, channels = frame.shape
            blob = preprocessing_image(frame)
            outputs = detect_objects(blob, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors, class_ids, classes, frame)
            video_writer.write(frame)
            cv2.imshow("Tracking Traffic", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            break
    video_writer.release()
    cap.release()