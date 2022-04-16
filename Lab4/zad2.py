import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#color
data_dir_color = os.path.join('obrazy_testowe', 'color')
images = sorted(os.listdir(data_dir_color))
img_path_color = os.path.join(data_dir_color, images[303785%len(images)])
image_color = cv2.imread(img_path_color)

# # Zadanie 2

image_equalized = cv2.cvtColor(image_color, cv2.COLOR_BGR2YUV)
original_histogram = cv2.calcHist([image_equalized], [0], None, [255], [0, 255])
image_equalized[:, :, 0] = cv2.equalizeHist(image_equalized[:, :, 0])
equalized_histogram = cv2.calcHist([image_equalized], [0], None, [255], [0, 255])
image_equalized = cv2.cvtColor(image_equalized, cv2.COLOR_YUV2BGR)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].plot(original_histogram.flatten())
axs[1].plot(equalized_histogram.flatten())
axs[0].set_xlim(0, 255)
axs[1].set_xlim(0, 255)
axs[0].set_title("oryginalny histogram")
axs[1].set_title("wyrownany histogram")
plt.show()
cv2.imwrite("equalized_image.png", image_equalized)