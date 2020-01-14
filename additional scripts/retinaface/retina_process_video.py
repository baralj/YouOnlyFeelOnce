import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

thresh = 0.8

im_scale = 1.0

scales = [im_scale]
flip = False


count = 1

gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
color = (0,0,255)

#img_path = "image.jpg"
#img = cv2.imread(img_path)
#rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

for file in os.listdir("videos/Validation_Set/"):
    vidcap = cv2.VideoCapture("videos/Validation_Set/" + file)

    create_directory("test_ann_val/" + file[:-4] + "/")
    if os.path.exists("annotations/Validation_Set/" + file[:-4] + "_left.txt"):
        with open("annotations/Validation_Set/" + file[:-4] + "_left.txt", "r") as f_ann_left, \
                  open("annotations/Validation_Set/" + file[:-4] + "_right.txt", "r") as f_ann_right:
            f_ann_left.readline()
            f_ann_right.readline()

            anns_left = list(map(int, filter(None, map(str.strip, f_ann_left.readlines()))))
            anns_right = list(map(int, filter(None, map(str.strip, f_ann_right.readlines()))))
            count_anns = 0

            success, image = vidcap.read()
            dims = image.shape
            count = 0
            while success and count_anns < len(anns_left):

                ann1 = anns_left[count_anns]
                ann2 = anns_right[count_anns]

                count_anns += 1

                if ann1 == -1 and ann2 == -1:
                    success, image = vidcap.read()
                    continue



                faces, _ = detector.detect(image, thresh, scales=scales, do_flip=flip)

                #bboxes = sorted(face_detector.predict(image, 0.5), key=lambda x: -x[4])
                if faces.shape[0] >= 2:
                    true_fc_l = faces[0].astype(float)
                    true_fc_r = faces[0].astype(float)

                    for fc in faces:
                        box1 = fc.astype(np.float)
                        # if box1[2] > true_fc[2]:
                        if box1[1] < dims[0] / 4.0 and box1[0] < true_fc_l[0]:
                            true_fc_l = box1
                        if box1[2] > true_fc_r[2]:
                            true_fc_r = box1

                    x_center1 = float(true_fc_l[0] + true_fc_l[2]) / (2 * float(dims[1]))
                    y_center1 = float(true_fc_l[1] + true_fc_l[3]) / (2 * float(dims[0]))
                    width1 = float(true_fc_l[2] - true_fc_l[0]) / float(dims[1])
                    height1 = float(true_fc_l[3] - true_fc_l[1]) / float(dims[0])

                    x_center2 = float(true_fc_r[0] + true_fc_r[2]) / (2 * float(dims[1]))
                    y_center2 = float(true_fc_r[1] + true_fc_r[3]) / (2 * float(dims[0]))
                    width2 = float(true_fc_r[2] - true_fc_r[0]) / float(dims[1])
                    height2 = float(true_fc_r[3] - true_fc_r[1]) / float(dims[0])

                    f = open("obj_val_corr/" + file[:-4] + "_" + ("%05d" % count) + ".txt", "w")

                    if ann1 != -1:
                        if ann1 == 0:
                            ann1 = 6
                        else:
                            ann1 = ann1 - 1

                        f.write(str(ann1) + " " + str(round(x_center1, 6)) + " " + str(round(y_center1, 6)) + " " +
                                str(round(width1, 6)) + " " + str(round(height1, 6)) + "\n")

                    if ann2 != -1:
                        if ann2 == 0:
                            ann2 = 6
                        else:
                            ann2 = ann2 - 1
                        f.write(str(ann2) + " " + str(round(x_center2, 6)) + " " + str(round(y_center2, 6)) + " " +
                                str(round(width2, 6)) + " " + str(round(height2, 6)) + "\n")

                    f.close()
                    ann_img = image.copy()
                    cv2.rectangle(ann_img, (int( true_fc_l[0]), int( true_fc_l[1])), (int( true_fc_l[2]), int( true_fc_l[3])), color, 2)
                    cv2.rectangle(ann_img, (int(true_fc_r[0]), int(true_fc_r[1])), (int(true_fc_r[2]), int(true_fc_r[3])), color, 2)
                    cv2.imwrite("obj_val_corr/" + file[:-4] + "_" + ("%05d" % count) + ".jpg", image)
                    cv2.imwrite("test_ann_val/" + file[:-4] + "/" + file[:-4] + "_" + ("%05d" % count)  + ".jpg", ann_img)
                    success, image = vidcap.read()
                    print('Read frame num: ', count)
                    count += 1

                elif faces.shape[0] >= 1:
                    f = open("obj_val_corr/" + file[:-4] + "_" + ("%05d" % count) + ".txt", "w")
                    box1 = faces[0].astype(np.float)

                    x_center1 = float(box1[0] + box1[2]) / (2 * float(dims[1]))
                    y_center1 = float(box1[1] + box1[3]) / (2 * float(dims[0]))
                    width1 = float(box1[2] - box1[0]) / float(dims[1])
                    height1 = float(box1[3] - box1[1]) / float(dims[0])

                    if ann1 != -1 and ann2 != -1:
                        if ann1 == 0:
                            ann1 = 6
                        else:
                            ann1 = ann1 - 1

                        if ann2 == 0:
                            ann2 = 6
                        else:
                            ann2 = ann2 - 1

                        if box1[0] < dims[1] / 2.0:
                            f.write(str(ann1) + " " + str(round(x_center1, 6)) + " " + str(round(y_center1, 6)) + " " +
                                    str(round(width1, 6)) + " " + str(round(height1, 6)) + "\n")
                        else:
                            f.write(str(ann2) + " " + str(round(x_center1, 6)) + " " + str(round(y_center1, 6)) + " " +
                                    str(round(width1, 6)) + " " + str(round(height1, 6)) + "\n")
                    elif ann1 == -1:
                        if ann2 == 0:
                            ann2 = 6
                        else:
                            ann2 = ann2 - 1
                        f.write(str(ann2) + " " + str(round(x_center1, 6)) + " " + str(round(y_center1, 6)) + " " +
                                str(round(width1, 6)) + " " + str(round(height1, 6)) + "\n")

                    else:
                        if ann1 == 0:
                            ann1 = 6
                        else:
                            ann1 = ann1 - 1
                        f.write(str(ann1) + " " + str(round(x_center1, 6)) + " " + str(round(y_center1, 6)) + " " +
                                str(round(width1, 6)) + " " + str(round(height1, 6)) + "\n")

                    f.close()
                    ann_img = image.copy()
                    cv2.rectangle(ann_img, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), color, 2)
                    cv2.imwrite("obj_val_corr/" + file[:-4] + "_" + ("%05d" % count) + ".jpg", image)
                    cv2.imwrite("test_ann_val/" + file[:-4] + "/" + file[:-4] + "_" + ("%05d" % count) + ".jpg", ann_img)
                    success, image = vidcap.read()
                    print('Read frame num: ', count)
                    count += 1
                else:
                    success, image = vidcap.read()
                    continue


    else:
        with open("annotations/Validation_Set/" + file[:-4] + ".txt", "r") as ann_one:
            ann_one.readline()
            success, image = vidcap.read()
            dims = image.shape
            count = 0
            anns = list(map(int, filter(None, map(str.strip, ann_one.readlines()))))
            count_anns = 0
            while success and count_anns < len(anns):
                ann = anns[count_anns]
                count_anns += 1

                if ann == -1:
                    success, image = vidcap.read()
                    continue

                if ann == 0:
                    ann = 6
                else:
                    ann = ann - 1

                faces, _ = detector.detect(image, thresh, scales=scales, do_flip=flip)

                #if faces.shape[0] >= 1:
                if faces.shape[0] == 1:
                    box1 = faces[0].astype(np.float)

                    x_center1 = float(box1[0] + box1[2]) / (2 * float(dims[1]))
                    y_center1 = float(box1[1] + box1[3]) / (2 * float(dims[0]))
                    width1 = float(box1[2] - box1[0]) / float(dims[1])
                    height1 = float(box1[3] - box1[1]) / float(dims[0])

                    f = open("obj_val_corr/" + file[:-4] + "_" + ("%05d" % count) + ".txt", "w")
                    f.write(str(ann) + " " + str(round(x_center1, 6)) + " " + str(round(y_center1, 6)) + " " +
                            str(round(width1, 6)) + " " + str(round(height1, 6)) + "\n")
                    f.close()

                    ann_img = image.copy()
                    cv2.rectangle(ann_img, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), color, 2)
                    cv2.imwrite("obj_val_corr/" + file[:-4] + "_" + ("%05d" % count) + ".jpg", image)
                    cv2.imwrite("test_ann_val/" + file[:-4] + "/" + file[:-4] + "_" + ("%05d" % count) + ".jpg", ann_img)
                    success, image = vidcap.read()
                    print('Read frame num: ', count)
                    count += 1

                elif faces.shape[0] > 1:
                    true_fc = faces[0].astype(float)
                    for fc in faces:
                        box1 = fc.astype(np.float)
                        #if box1[2] > true_fc[2]:
                        if box1[2] > true_fc[2]:
                            true_fc = box1


                    x_center1 = float(true_fc[0] + true_fc[2]) / (2 * float(dims[1]))
                    y_center1 = float(true_fc[1] + true_fc[3]) / (2 * float(dims[0]))
                    width1 = float(true_fc[2] - true_fc[0]) / float(dims[1])
                    height1 = float(true_fc[3] - true_fc[1]) / float(dims[0])

                    f = open("obj_val_corr/" + file[:-4] + "_" + ("%05d" % count) + ".txt", "w")
                    f.write(str(ann) + " " + str(round(x_center1, 6)) + " " + str(round(y_center1, 6)) + " " +
                            str(round(width1, 6)) + " " + str(round(height1, 6)) + "\n")
                    f.close()

                    ann_img = image.copy()
                    cv2.rectangle(ann_img, (int(true_fc[0]), int(true_fc[1])), (int(true_fc[2]), int(true_fc[3])), color, 2)
                    cv2.imwrite("obj_val_corr/" + file[:-4] + "_" + ("%05d" % count) + ".jpg", image)
                    cv2.imwrite("test_ann_val/" + file[:-4] + "/" + file[:-4] + "_" + ("%05d" % count) + ".jpg",
                                ann_img)
                    success, image = vidcap.read()
                    print('Read frame num: ', count)
                    count += 1

                else:
                    success, image = vidcap.read()

