import os
import cv2

img_path = "C:\\Users\\adria\\Documents\\GitHub\\leaf-count-cv\\data\\0028.jpg"
output_dir = "C:\\Users\\adria\\Documents\\GitHub\\leaf-count-cv\\leafs"
leaf_number = 280

def cutleaf(img_path, output_dir, leaf_number):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w*h < 1000 or w*h > 2073599:     # 1920x1080
            continue

        leaf = img[y:y+h, x:x+w]

        output_path = os.path.join(output_dir, str(leaf_number)+".jpg")
        cv2.imwrite(output_path, leaf)

        leaf_number += 1

cutleaf(img_path, output_dir, leaf_number)