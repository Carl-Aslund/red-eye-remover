import numpy as np
import cv2

IMG_SRC = "red_eye_couple.jpg"

def fillHoles(mask):
    """Fill potential holes in a generated mask."""
    maskFloodFill = mask.copy()
    h, w = maskFloodFill.shape[:2]
    maskTemp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(maskFloodFill, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(maskFloodFill)
    return mask2 | mask

# Read the image
img = cv2.imread("test_img/"+IMG_SRC, cv2.IMREAD_COLOR)

# Create an image copy
imgOut = img.copy()

# Load HAAR cascades
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Load camera feed
#cap = cv2.VideoCapture(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    roiGray = gray[y:y+h, x:x+w]
    roiColor = img[y:y+h, x:x+w]

    eyes = eyesCascade.detectMultiScale(roiGray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roiColor, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        # Extract the individual eye from the larger image
        eye = img[y:y+h, x:x+w]

        # Split colors into RGB
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]

        # Combine green and blue
        bg = cv2.add(b, g)

        # Red eye mask
        mask = (r > 150) & (r > (bg)*1.5)

        # Convert the mask format
        mask = mask.astype(np.uint8)*255

        # Clean up the mask by filling holes and dilating
        mask = fillHoles(mask)
        mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

        # Average the blue and green values
        mean = bg//2
        mask = mask.astype(np.bool)[:, :, np.newaxis]
        mean = mean[:, :, np.newaxis]

        # Copy the eye from the original image
        eyeOut = eye.copy()

        # Copy an averaged eye to the output
        np.copyto(eyeOut, mean, where=mask)

        # Merge the new eye into the output image
        img[y:y+h, x:x+w, :] = eyeOut

while True:
    cv2.imshow("img",img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

"""
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        roiGray = gray[y:y+h, x:x+w]
        roiColor = img[y:y+h, x:x+w]

        eyes = eyesCascade.detectMultiScale(roiGray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roiColor, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

    cv2.imshow("img",img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
"""

"""
# Detect eyes
eyes = eyesCascade.detectMultiScale(img,scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))

if len(eyes) != 2:
    print("ERROR:", len(eyes), "found!")

for (x,y,w,h) in eyes:
    # Extract the individual eye from the larger image
    eye = img[y:y+h, x:x+w]

    # Split colors into RGB
    b = eye[:, :, 0]
    g = eye[:, :, 1]
    r = eye[:, :, 2]

    # Combine green and blue
    bg = cv2.add(b, g)

    # Red eye mask
    mask = (r > 150) & (r > bg)

    # Convert the mask format
    mask = mask.astype(np.uint8)*255

    # Clean up the mask by filling holes and dilating
    mask = fillHoles(mask)
    mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

    # Average the blue and green values
    mean = bg//2
    mask = mask.astype(np.bool)[:, :, np.newaxis]
    mean = mean[:, :, np.newaxis]

    # Copy the eye from the original image
    eyeOut = eye.copy()

    # Copy an averaged eye to the output
    np.copyto(eyeOut, mean, where=mask)

    # Merge the new eye into the output image
    imgOut[y:y+h, x:x+w, :] = eyeOut

cv2.imshow('image',imgOut)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
