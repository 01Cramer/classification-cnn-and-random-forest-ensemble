import json
from pathlib import Path
from typing import Dict
from matplotlib import pyplot as plt
import click
import cv2
import numpy as np
from tqdm import tqdm
default_cmap = plt.cm.coolwarm
import imageio

img_path = "C:\\Users\\adria\\Documents\\GitHub\\leaf-count-cv\\data\\0001.jpg"

def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        A dictionary with the number of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # TODO: Implement detection method.

    aspen = 0
    birch = 0
    hazel = 0
    maple = 0
    oak = 0

    return {'aspen': aspen, 'birch': birch, 'hazel': hazel, 'maple': maple, 'oak': oak}


img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

ret, binary = cv2.threshold(img,230,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

number_of_leafs = 0
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    if w*h < 100 or w*h > 2073599: # 1920x1080
        continue
    print(x, y, w, h)
    number_of_leafs += 1
    # draw the bounding boxes
    cv2.rectangle(binary, (x, y), (x + w, y + h), (0, 255, 0), 2)


print("Number of leafs: ", number_of_leafs)
plt.subplot(1, 1, 1)
plt.imshow(binary, cmap='gray')
plt.show()
