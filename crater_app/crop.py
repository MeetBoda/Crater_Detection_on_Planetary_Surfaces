import cv2

img = cv2.imread('thm_dir_N-30_000.png', cv2.IMREAD_GRAYSCALE)

h, w = img.shape[:2]

crop_size = 1024

start_x = (w - crop_size) // 2
start_y = (h - crop_size) // 2

cropped = img[start_y:start_y+crop_size, start_x:start_x+crop_size]

cv2.imwrite('input1024.png', cropped)