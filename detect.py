import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm


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