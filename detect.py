import tensorflow as tf
import cv2
import os
import numpy as np
import json
from pathlib import Path
from typing import Dict
import click
from tqdm import tqdm
from joblib import load

def prepare(leaf_path):
    leaf_size = 128
    leaf_img = cv2.imread(leaf_path, cv2.IMREAD_COLOR)
    leaf_img = cv2.resize(leaf_img, (leaf_size, leaf_size))
    leaf_img = leaf_img/255.0
    leaf_img = np.array(leaf_img).reshape(-1, leaf_size, leaf_size, 3)
    return leaf_img


def detect(img_path: str) -> Dict[str, int]:
    output_dir = "leafs"
    os.makedirs(output_dir) # Creates a folder where single leaves (from one image) will be stored
    leaf_number = 0

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

#------------------------------------------------- NEURAL NET MODEL ---------------------------------------------------#
    model = tf.keras.models.load_model('feature_extractor_model.h5')

    clf = load('RF_model.joblib')


    aspen = 0
    birch = 0
    hazel = 0
    maple = 0
    oak = 0

    leafs_to_classify = os.listdir(output_dir)

    for leaf in leafs_to_classify:
        leaf_path = os.path.join(output_dir, leaf)
        leaf_transformed = prepare(leaf_path)

        leaf_transformed = model.predict(leaf_transformed)

        classification = clf.predict(leaf_transformed)

        if classification == 0:
            aspen += 1
        elif classification == 1:
            birch += 1
        elif classification == 2:
            hazel += 1
        elif classification == 3:
            maple += 1
        else:
            oak += 1

    # Clears directory before removing it
    file_list = os.listdir(output_dir)
    for file_name in file_list:
        file_path = os.path.join(output_dir, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Can't delete file {file_path}: {e}")

    os.rmdir(output_dir)
    return {'aspen': aspen, 'birch': birch, 'hazel': hazel, 'maple': maple, 'oak': oak}

@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)

def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        leaves = detect(str(img_path))
        results[img_path.name] = leaves

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()