import os
import random
import sys
import cv2
from shutil import copyfile

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
#data_size = 1400
#loop_index = 0
create_directory("obj")
filenames_all = []
emo_num = [0, 0, 0, 0, 0, 0, 0]
i = 0
with open("label/label.lst", "r") as files:
    for row in files:
        data = row.strip().split()
        if data:
            if os.path.exists("image/origin/" + data[0]):
                if data[1] == "0" and float(data[6]) > 50:
                    img = cv2.imread("image/origin/" + data[0])
                    dims = img.shape
                    x_center = (float(data[3]) + float(data[4])) / (2.0 * float(dims[1]))
                    y_center = (float(data[2]) + float(data[5])) / (2.0 * float(dims[0]))
                    width = (float(data[4]) - float(data[3])) / float(dims[1])
                    height = (float(data[5]) - float(data[2])) / float(dims[0])
                              
                    f = open("E:/Programs/Darknet/Release/data/obj/" + data[0][:-3] + "txt", "w")
                    f.write(data[7] + " " + str(round(x_center, 6)) + " " + str(round(y_center, 6)) + " " + 
                        str(round(width, 6)) + " " + str(round(height, 6)) + "\n")
                    f.close()
                    
                    cv2.rectangle(img,(int(data[3]),int(data[2])),(int(data[4]),int(data[5])),(0,255,0),3)
                    cv2.imwrite('annotated/' + data[0], img)
                    print(data[0] + " " + data[7])
                    #if i == 100:
                    #    break
                    #i += 1
                    #continue
            
                #print("Processing " + data[0])
                
                
                img = cv2.imread("image/origin/" + data[0])
                dims = img.shape
                x_center = (float(data[3]) + float(data[4])) / (2.0 * float(dims[1]))
                y_center = (float(data[2]) + float(data[5])) / (2.0 * float(dims[0]))
                width = (float(data[4]) - float(data[3])) / float(dims[1])
                height = (float(data[5]) - float(data[2])) / float(dims[0])
                          
                f = open("E:/Programs/Darknet/Release/data/obj/" + data[0][:-3] + "txt", "w")
                f.write(data[7] + " " + str(round(x_center, 6)) + " " + str(round(y_center, 6)) + " " + 
                    str(round(width, 6)) + " " + str(round(height, 6)) + "\n")
                f.close()
                
                #cv2.rectangle(img,(int(data[3]),int(data[2])),(int(data[4]),int(data[5])),(0,255,0),3)
                #cv2.imwrite('annotated/' + data[0], img)
                #cv2.waitKey()
                
                copyfile("image/origin/" + data[0], "E:/Programs/Darknet/Release/data/obj/" + data[0])
                
                filenames_all.append((data[0], int(data[7])))
                emo_num[int(data[7])] += 1
                
     
random.shuffle(filenames_all)
print(len(filenames_all))
print(emo_num)
'''
emo_test_num = list(map(lambda x: int(x * 0.1), emo_num))
emo_valid_num = list(map(lambda x: int(x * 0.1), emo_num))
print(emo_test_num)

with open("train.txt", "w") as f_train, open("test.txt", "w") as f_test, open("valid.txt", "w") as f_valid:  
    for file, emo in filenames_all:
        if emo_test_num[emo] > 0:
            f_test.write("data_ExpW/obj/" + file + "\n")
            emo_test_num[emo] -= 1
        elif emo_valid_num[emo] > 0:
            f_valid.write("data_ExpW/obj/" + file + "\n")
            emo_valid_num[emo] -= 1
        else:
            f_train.write("data_ExpW/obj/" + file + "\n")
    
'''

loop_index = 0
with open("train.txt", "w") as f_train, open("test.txt", "w") as f_test, open("valid.txt", "w") as f_valid:
    data_sampsize = int(0.1 * len(filenames_all))
    
    for file in filenames_all:
        if loop_index <= data_sampsize:
            f_test.write("data_ExpW/obj/" + file + "\n")
        elif loop_index <= 2 * data_sampsize:
            f_valid.write("data_ExpW/obj/" + file + "\n")
        else:
            f_train.write("data_ExpW/obj/" + file + "\n")
        loop_index += 1

