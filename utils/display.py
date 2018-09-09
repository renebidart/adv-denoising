import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

#########   DISPLAYING UTILS
def load_image(img_path, size=32):
    """Load jpg image into format for Foolbox"""
    img = np.asarray(Image.open(img_path).convert('RGB').resize((size, size), Image.ANTIALIAS))
    img = np.moveaxis(img, 2, 0) / 255.
    return img.astype('float32')

def show_img(image, adversarial):
    """Ajdust foolbox format to matplotlib format and olot"""
    image = np.moveaxis(image, 0, 2)
    adversarial = np.moveaxis(adversarial, 0, 2)
    difference = adversarial - image
    
    plt.figure(figsize=(10,10))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow( image)  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial')
    plt.imshow( adversarial)  # ::-1 to convert BGR to RGB
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')
    plt.show()