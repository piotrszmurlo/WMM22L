import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# color
data_dir_color = os.path.join('obrazy_testowe', 'color')
images = sorted(os.listdir(data_dir_color))
img_path_color = os.path.join(data_dir_color, images[303785%len(images)])
image_color = cv2.imread(img_path_color)

# Zadanie 3
laplacian_image = cv2.Laplacian(image_color, cv2.CV_8U) 
cv2.imwrite("laplacian_image.png", laplacian_image)
for id, weight in enumerate([0.5, 1.0, 1.5, 2.0]):
    result = cv2.addWeighted(image_color, 1, laplacian_image, -weight, 0)
    cv2.imwrite(f"laplacian_image_w{id}.png", result)