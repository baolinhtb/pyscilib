'''
This code is based on a simple algorithm for cleaning up scan images
visit https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699
for more detail.
'''

import cv2, sys
import numpy as np
from PIL import Image

def remove_shadow_img(img):
	# img = cv2.imread('shadows.png', -1)

	rgb_planes = cv2.split(img)

	result_planes = []
	result_norm_planes = []
	for plane in rgb_planes:
	    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
	    bg_img = cv2.medianBlur(dilated_img, 21)
	    diff_img = 255 - cv2.absdiff(plane, bg_img)
	    norm_img = diff_img.copy()
	    norm_img = cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
	    	dtype=cv2.CV_8UC1)
	    # result_planes.append(diff_img)
	    result_norm_planes.append(norm_img)

	# result = cv2.merge(result_planes)
	result_norm = cv2.merge(result_norm_planes)
	return result_norm

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)

    return img.point(contrast)


def increase_contrast(im):
    pil_im = Image.fromarray(im)
    return np.array(change_contrast(pil_im, 5))


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def equalize_hist(img):
    for c in range(0, 2):
       img[:,:,c] = cv2.equalizeHist(img[:,:,c])
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


if __name__ == '__main__':
	img = cv2.imread(sys.argv[1], -1)
	removed_shadow_img = adjust_gamma(increase_contrast(remove_shadow_img(img)), 0.4)
	cv2.imwrite(sys.argv[2], removed_shadow_img)
