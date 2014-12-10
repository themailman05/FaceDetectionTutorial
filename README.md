ISAT 344 OPENCV FACE-DETECTION TUTORIAL
==================

Inside this repository there are a number of scripts that have different functions.

Images for testing different algorithms are in the TestImages folder.

Open-source haar cascades for different patterns are located in haarcascades

## Face detection on static image

'python face_detect.py YOUR_IMAGE_HERE haarcascade_frontalface_default.xml'

## With alternative facial dataset

'python face_detect.py YOUR_IMAGE_HERE haarcascade_frontalface_alt.xml'


## Webcam input

'python webcam.py haarcascade_frontalface_default.xml'

## Multiple objects via Webcam

'python webcamfun.py face.xml eye.xml palm.xml'

