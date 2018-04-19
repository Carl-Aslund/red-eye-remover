from remover import remove_redeye
import numpy as np
import cv2

from os import listdir
from os.path import isfile, join

IMG_FOLDER = "test_img/"
FIXED_FOLDER = "fixed_img/"

images = [f for f in listdir(IMG_FOLDER) if isfile(join(IMG_FOLDER, f))]

for image in images:
    img = cv2.imread(IMG_FOLDER + image, cv2.IMREAD_COLOR)
    cv2.imwrite(FIXED_FOLDER + image, remove_redeye(img))

