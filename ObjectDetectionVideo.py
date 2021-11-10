import cv2
import numpy as np

# Load yolo
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print(output_layers)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    height, width, channels = img.shape
    return img, height, width, channels


def display_blob(blob):
    '''
      Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
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


def draw_labels(boxes, confs, colors, class_ids, classes, img, video):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    print(boxes)
    font = cv2.QT_FONT_BLACK
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            cv2.putText(img, str(round((confs[0]*100), 2))+"%", (x + w//2, y - 5), font, 1, color, 1)
    video.write(img)


if __name__ == '__main__':
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture("videos/Driving_Downtown_New_York_City.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter("output.avi", fourcc, 20, (640, 480))
    while True:
        crt, frame = cap.read()
        if crt:
            height, width, channels = frame.shape
            blob, outputs = detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors, class_ids, classes, frame, video)
            # cv2.imshow("Transportation", frame)
            # key = cv2.waitKey(1)
            # if key == 27:
            #     break
        else:
            break
    cap.release()