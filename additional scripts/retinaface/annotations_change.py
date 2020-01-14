import glob
import os

path = 'obj_train/'

for filename in glob.glob(os.path.join(path, '*.txt')):
	with open("obj_train" + filename[9:], "r") as inFile, open("E:/Programs/Darknet/Release/data_ExpW2/obj" + filename[9:], "w") as outFile:
		for line in inFile:
			ann = int(line[0])
			if ann == 0:
				outFile.write("6" + line[1:])
			else:
				outFile.write(str(ann-1) + line[1:])