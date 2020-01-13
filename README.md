## You Only Feel Once

This is a repository containing code and report for third seminar for class Image Based Biometrics at Faculty of Computer and Information Science, University of Ljubljana. For this seminar I trained Yolov3-tiny network for facial expression detection. Training was done by Darknet framework and then the model was converted to format that can be used by Tensorflow library.

To run inference code on a single image, call program with the path to image as argument like that:

```
python demo.py samples/82-25-854x480_00529.jpg
```

The output image with predictions as bounding boxes will be written to *predictions.jpg*.



To run inference code on camera stream, call program without argument, like that:

```
python demo.py
```

------

**Important notice**

The code was written using Tensorflow 1.14. It will probably work on other versions of Tensorflow before 2.0 but due to major changes it won't work in Tensorflow >= 2.0.

Two other Python libraries that are needed are OpenCV, Pillow and Numpy.

