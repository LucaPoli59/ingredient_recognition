import cv2
import pandas as pd
import matplotlib.pyplot as plt

from settings.config import *

shapes = []
names = []
for category in os.listdir(YUMMLY_PATH):
    for image in os.listdir(os.path.join(YUMMLY_PATH, category))[:3]:
        img = cv2.imread(os.path.join(YUMMLY_PATH, category, image))
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