import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
firstPath = sys.argv[2]
secondPath = sys.argv[3]


# Create the haar cascade
faceCascade = cv2.CascadeClassifier(firstPath)
eyeCascade = cv2.CascadeClassifier(secondPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# Detect Eyes in the image
eyes = eyeCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Draw a rectangle around the eyes
for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Faces & Eyes found", image)
cv2.waitKey(0)
