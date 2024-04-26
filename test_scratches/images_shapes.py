import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from config import *

shapes = []
names = []
for category in os.listdir(IMAGES_PATH):
    for image in os.listdir(os.path.join(IMAGES_PATH, category))[:3]:
        img = cv2.imread(os.path.join(IMAGES_PATH, category, image))
        shapes.append(list(img.shape))
        names.append(category + "_" + image.strip('.jpg'))

shape_df = pd.DataFrame(shapes, columns=['height', 'width', 'channels'], index=names)
print(shape_df.describe())
print("\n", shape_df.mode())
print("\n", len(shape_df))
shape_df.plot(kind="hist", bins=20, alpha=0.5, title="Image Shapes", subplots=True)
plt.show()



"""
Da questo grafico si nota che la maggior parte delle immagini ha una dimensione di 240x360x3, le immagini diverse saranno fixate con il padding 
"""