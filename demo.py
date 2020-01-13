import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import sys


class Cluster:
    def __init__(self, t, b, l, r, score, emot):
        self.top = t
        self.bottom = b
        self.left = l
        self.right = r
        self.score = score
        self.emot = emot
        self.top_a = np.array([t])
        self.bottom_a = np.array([b])
        self.left_a = np.array([l])
        self.right_a = np.array([r])


def suppression(boxes):
    clusters = []
    for box in boxes:
        in_cluster = False
        for c in clusters:
            for idx in range(len(c.top_a)):
                int_l = max(box[0], c.left_a[idx])
                int_r = min(box[2], c.right_a[idx])
                int_t = max(box[1], c.top_a[idx])
                int_b = min(box[3], c.bottom_a[idx])

                if int_l < int_r and int_t < int_b:
                    area = (int_r - int_l) * (int_b - int_t)
                    overlap = area / float((c.bottom_a[idx]-c.top_a[idx])*(c.right_a[idx]-c.left_a[idx]) +
                         (box[3]-box[1])*(box[2]-box[0]) - area)
                    if overlap > 0.1:
                        c.top_a = np.append(c.top_a, box[1])
                        c.bottom_a = np.append(c.bottom_a, box[3])
                        c.left_a = np.append(c.left_a, box[2])
                        c.right_a = np.append(c.right_a, box[0])
                        if box.score > c.score:
                            c.top = box[1]
                            c.bottom = box[3]
                            c.left = box[0]
                            c.right = box[2]
                            c.score = box.score
                        in_cluster = True
                        break
            if in_cluster:
                break

        if not in_cluster:
            new_cluster = Cluster(box.t,box.b,box.l, box.r, box.score)
            clusters.append(new_cluster)

    return clusters


def extract_boxes(predictions, confidence_threshold):
    conf_mask = np.expand_dims(
        (predictions[:, :, 4] > confidence_threshold), -1)
    predictions = predictions * conf_mask

    result = []
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        clusters = []
        for box in image_pred:
            in_cluster = False
            for c in clusters:
                for idx in range(len(c.top_a)):
                    int_l = max(box[0], c.left_a[idx])
                    int_r = min(box[2], c.right_a[idx])
                    int_t = max(box[1], c.top_a[idx])
                    int_b = min(box[3], c.bottom_a[idx])

                    if int_l < int_r and int_t < int_b:
                        area = (int_r - int_l) * (int_b - int_t)
                        overlap = area / float((c.bottom_a[idx] - c.top_a[idx]) * (c.right_a[idx] - c.left_a[idx]) +
                                               (box[3] - box[1]) * (box[2] - box[0]) - area)
                        if overlap > 0.1:
                            c.top_a = np.append(c.top_a, box[1])
                            c.bottom_a = np.append(c.bottom_a, box[3])
                            c.left_a = np.append(c.left_a, box[2])
                            c.right_a = np.append(c.right_a, box[0])
                            if box[4] > c.score:
                                c.top = box[1]
                                c.bottom = box[3]
                                c.left = box[0]
                                c.right = box[2]
                                c.score = box[4]
                                c.emot = np.argmax(box[5:])
                            in_cluster = True
                            break
                if in_cluster:
                    break

            if not in_cluster:
                new_cluster = Cluster(box[1], box[3], box[0], box[2], box[4], np.argmax(box[5:]))
                clusters.append(new_cluster)

        result.append([])
        for c in clusters:
            result[-1].append([c.left, c.top, c.right, c.bottom, c.emot])
    return result

def process_image(input_img):
    classes = {
        0: "Anger",
        1: 'Disgust',
        2: 'Fear',
        3: 'Happiness',
        4: 'Sadness',
        5: 'Surprise',
        6: 'Neutral'
    }

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    with tf.gfile.GFile("frozen_fer_model.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        boxes = graph.get_tensor_by_name("output_boxes:0")
        inputs = graph.get_tensor_by_name("inputs:0")

    with tf.Session(graph=graph, config=config) as sess:
        img = Image.open(input_img)
        img_orig = np.array(img)
        (w_orig, h_orig) = img.size
        img.thumbnail((416, 416), Image.ANTIALIAS)
        (w, h) = img.size
        img_np = np.zeros((416, 416, 3), np.uint8)
        cx, cy = (416 - img.size[0]) // 2, (416 - img.size[1]) // 2
        img_np[cy:img.size[1] + cy, cx:cx + img.size[0]] = np.array(img)

        detected_boxes = sess.run(
            boxes, feed_dict={inputs: [img_np]})

        filtered_boxes = extract_boxes(detected_boxes, 0.1)
        color = (0, 255, 0)

        for box in filtered_boxes[0]:

            if w == 416:
                box[0] *= w_orig / w
                box[2] *= w_orig / w
            else:
                box[0] -= cx
                box[0] *= w_orig / w
                box[2] -= cx
                box[2] *= w_orig / w

            if h == 416:
                box[1] *= h_orig / h
                box[3] *= h_orig / h
            else:
                box[1] -= cy
                box[1] *= h_orig / h
                box[3] -= cy
                box[3] *= h_orig / h

            img_orig = cv2.rectangle(img_orig, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(img_orig, classes[box[4]], (int(box[0]) - 2, int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)
        img_bgr = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
        cv2.imwrite("predictions.jpg", img_bgr)
        print("Found " + str(len(filtered_boxes[0])) + " predictions. Image written to predictions.jpg")

def process_video():
    classes = {
        0: "Anger",
        1: 'Disgust',
        2: 'Fear',
        3: 'Happiness',
        4: 'Sadness',
        5: 'Surprise',
        6: 'Neutral'
    }

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    with tf.gfile.GFile("frozen_fer_model.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        boxes = graph.get_tensor_by_name("output_boxes:0")
        inputs = graph.get_tensor_by_name("inputs:0")

    cam = cv2.VideoCapture(0)
    ret_val, img = cam.read()
    if not ret_val:
        return
    else:
        with tf.Session(graph=graph, config=config) as sess:
            while ret_val:
                img_orig = np.array(img)
                img_pil = Image.fromarray(img)
                (w_orig, h_orig) = img_pil.size
                img_pil.thumbnail((416, 416), Image.ANTIALIAS)
                (w, h) = img_pil.size
                img_np = np.zeros((416, 416, 3), np.uint8)
                cx, cy = (416 - img_pil.size[0]) // 2, (416 - img_pil.size[1]) // 2
                img_np[cy:img_pil.size[1] + cy, cx:cx + img_pil.size[0]] = np.array(img_pil)

                detected_boxes = sess.run(
                    boxes, feed_dict={inputs: [img_np]})

                filtered_boxes = extract_boxes(detected_boxes, 0.1)
                color = (0, 255, 0)

                for box in filtered_boxes[0]:

                    if w == 416:
                        box[0] *= w_orig / w
                        box[2] *= w_orig / w
                    else:
                        box[0] -= cx
                        box[0] *= w_orig / w
                        box[2] -= cx
                        box[2] *= w_orig / w

                    if h == 416:
                        box[1] *= h_orig / h
                        box[3] *= h_orig / h
                    else:
                        box[1] -= cy
                        box[1] *= h_orig / h
                        box[3] -= cy
                        box[3] *= h_orig / h

                    img_orig = cv2.rectangle(img_orig, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(img_orig, classes[box[4]], (int(box[0])-2, int(box[1])-2), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 1)

                cv2.imshow('Prediction', img_orig)
                cv2.waitKey(10)
                ret_val, img = cam.read()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        process_video()
    else:
        process_image(sys.argv[1])