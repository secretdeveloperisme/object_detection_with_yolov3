import cv2
import numpy as np

cap = cv2.VideoCapture("videos/Driving_Downtown_New_York_City.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# video_writer = cv2.VideoWriter("output/output.mp4", fourcc, 10.0, (640, 360))


def load_yolo():
    """
    load yolo model and yolo configuration to module deep network neural opencv2
    load class names
    :return:
    network, 80 class names, 80 colors, output layer names
    """
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    classes = []
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print((layers_names))
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def display_blob(blob):
    """
      Three images each for RED, GREEN, BLUE channel
    """
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def preprocessing_image(img):
    """
    preprocessing raw image to suitable for input yolo
    :param img:
    :return: image blob (binary large object)
    """
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=(320, 320), swapRB=True, crop=False)
    return blob


def detect_objects(blob, net, outputLayers):
    """
    transfer input blob to yolo network to detect object and get output from yolo network
    :param blob:
    :param net:
    :param outputLayers:
    :return: all detected bounding box with probability class
    """
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return outputs


def get_box_dimensions(outputs, height, width):
    """
    filter bounding box with confident equal 50 %
    convert detected bounding box to attributes :
       x begin horizontal address
       y begin vertical address
       w width of bounding box
       h height of bounding box
    and class probability
    :param outputs:
    :param height:
    :param width:
    :return:
    """
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
    """
    use non maximum supression to select best bounding box out of many overlapping bounding boxes
    :param boxes:
    :param confs:
    :return:
    """
    indexes = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.5, nms_threshold=0.4)
    return indexes


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    """
    draw bounding box of object corresponding color
    draw name of object
    draw class probability
    count each type of class and draw on the screen
    :param boxes:
    :param confs:
    :param colors:
    :param class_ids:
    :param classes:
    :param img:
    :return:
    """
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
            cv2.imshow("Real Traffic", frame)
            blob = preprocessing_image(frame)
            outputs = detect_objects(blob, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors, class_ids, classes, frame)
            # video_writer.write(frame)
            cv2.imshow("Tracking Traffic", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            break
    # video_writer.release()
    cap.release()
