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
        document_contour = approx
        break
def reorder_points(pts):
    pts = pts.reshape((4, 2))
    reordered = np.zeros((4, 2), dtype=np.float32)
    s = np.sum(pts, axis=1)
    d = np.diff(pts, axis=1)
    reordered[0] = pts[np.argmin(s)]
    reordered[2] = pts[np.argmax(s)]
    reordered[3] = pts[np.argmax(d)]
    reordered[1] = pts[np.argmin(d)]
    return reordered
width, height = 600, 800
pts1 = reorder_points(document_contour)
pts2 = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(original, matrix, (width, height))
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
scanned = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow("Scanned", scanned)
cv2.imshow("Original", original)    
cv2.waitKey(0)
cv2.destroyAllWindows()

