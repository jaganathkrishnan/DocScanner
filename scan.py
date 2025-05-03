import cv2
import numpy as np
image = cv2.imread("test.jpg")      #upload the image in the same directory as this file and rename it to test.jpg
original = image.copy()
image = cv2.resize(image, (800,800))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 75, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        screen_cnt = approx
        break
def reorder(pts):
    pts = pts.reshape((4, 2))
    

