#importing the packages
from skimage.filters import threshold_local
from transform import four_point_transform
import numpy as np
import argparse
import cv2
import imutils


# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())

# Load the image and check if the file is read correctly
image = cv2.imread(args["image"])
if image is None:
    print("Error: Could not load image. Please check the path.")
    exit(0)

# Resize the image
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Show original image and edge detected image
print("Step 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding the contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# Loop over the contours
for c in cnts:
    # Approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # If the approximated contour has 4 points, then we can assume it's the document
    if len(approx) == 4:
        screenCnt = approx
        break

# Show the contour of the paper
print("Step 2: Find the contour of the paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply 4-point transform to obtain a top-down view of the image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Convert the warped image to grayscale
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

# Show the original and scanned images
print("Step 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()
