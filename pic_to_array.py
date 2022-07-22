import cv2 as cv
import sys
import numpy as np

img=cv.imread('sudoku.jpeg',cv.IMREAD_GRAYSCALE)
img=cv.resize(img,(252,252))
cv.namedWindow("Display window")
images=[]
print(img.shape)
for i in range(9):
	for j in range(9):
		images.append(img[28*i:28*i+28,28*j:28*j+28])
cv.imshow("Display window",images[1])
if cv.waitKey(0) & 0xFF :
	exit 
