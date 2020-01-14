import os

with open("train_new.txt", "w") as f_train, open("test_new.txt", "w") as f_test:
	for file in os.listdir("./obj_train_corr"):
		if file.endswith(".jpg"):
			f_train.write("data/obj/" + file + "\n")
			
	for file in os.listdir("./obj_val_corr"):
		if file.endswith(".jpg"):
			f_test.write("data/obj/" + file + "\n")
			