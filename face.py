# import the necessary packages
from __future__ import print_function
import cv2
 
# load the image and convert it to grayscalei
image = cv2.imread("small_00.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)


#Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.35,
    minNeighbors=2,
    minSize=(3, 3),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


# draw the keypoints and show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)