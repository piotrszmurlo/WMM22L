import cv2
import numpy as np
import os

#color
data_dir_color = os.path.join('obrazy_testowe', 'color')
images = sorted(os.listdir(data_dir_color))
img_path_color = os.path.join(data_dir_color, images[303785%len(images)])
image_color = cv2.imread(img_path_color)

#color_inoise1
data_dir_inoise1 = os.path.join('obrazy_testowe', 'color_inoise1')
images = sorted(os.listdir(data_dir_inoise1))
img_path_inoise1 = os.path.join(data_dir_inoise1, images[303785%len(images)])
image_inoise1 = cv2.imread(img_path_inoise1)

#color_noise gauss
data_dir_noise = os.path.join('obrazy_testowe', 'color_noise')
images = sorted(os.listdir(data_dir_noise))
img_path_noise = os.path.join(data_dir_noise, images[303785%len(images)])
image_noise = cv2.imread(img_path_noise)

# Zadanie 1

def calcPSNR(img1, img2):
    imax = 255.**2  ### zakładana wartość pikseli z przedziału [0, 255]
    ##### w różnicy obrazów istotne są wartości ujemne, dlatego img1 konwertowany do typu np.float64 (liczby rzeczywiste) aby nie ograniczać wyniku do przedziału [0, 255]
    mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size  ### img1.size - liczba elementów w img1, ==img1.shape[0]*img1.shape[1] dla obrazów mono, ==img1.shape[0]*img1.shape[1]*img1.shape[2] dla obrazów barwnych
    return 10.0*np.log10(imax/mse)

psnr = [[], [], [], []]

for i in (3, 5, 7): # przetwarzanie obrazu filtrem Gaussa zaszumionego szumem gaussowskim
    g_gauss_blurred_img = cv2.GaussianBlur(image_noise, (i, i), 0)
    psnr[0].append(calcPSNR(image_color, g_gauss_blurred_img))
    cv2.imwrite(f"out_gauss_gauss_{i}x{i}.png", g_gauss_blurred_img)

for i in (3, 5, 7): # przetwarzanie obrazu filtrem Gaussa zaszumionego szumem impulsowym
    i_gauss_blurred_img = cv2.GaussianBlur(image_inoise1, (i, i), 0)
    psnr[1].append(calcPSNR(image_color, i_gauss_blurred_img))
    cv2.imwrite(f"out_inoise1_gauss_{i}x{i}.png", i_gauss_blurred_img)

for i in (3, 5, 7): # przetwarzanie obrazu filtrem medianowym zaszumionego szumem gaussowskim
    g_median_blurred_img = cv2.medianBlur(image_noise, i)
    psnr[2].append(calcPSNR(image_color, g_median_blurred_img))
    cv2.imwrite(f"out_gauss_median_{i}x{i}.png", g_median_blurred_img)

for i in (3, 5, 7): # przetwarzanie obrazu filtrem medianowym zaszumionego szumem impulsowym
    i_median_blurred_img = cv2.medianBlur(image_inoise1, i)
    psnr[3].append(calcPSNR(image_color, i_median_blurred_img))
    cv2.imwrite(f"out_inoise1_median_{i}x{i}.png", i_median_blurred_img)

for i in range(4):
    print(psnr[i])
