import remover

IMG_SRC = "red_eye_photomag.jpg"

while True:
    # Read the image
    img = cv2.imread("test_img/"+IMG_SRC, cv2.IMREAD_COLOR)
    cv2.imshow("img",remove_redeye(img))
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
