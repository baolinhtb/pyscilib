'''
This code is based on a simple algorithm for cleaning up scan images
visit https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699
for more detail.
'''

import cv2, sys
import numpy as np

def remove_shadow_img(img):
	# img = cv2.imread('shadows.png', -1)

	rgb_planes = cv2.split(img)

	result_planes = []
	result_norm_planes = []
	for plane in rgb_planes:
	    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
	    bg_img = cv2.medianBlur(dilated_img, 21)
	    diff_img = 255 - cv2.absdiff(plane, bg_img)
	    norm_img = cv2.normalize(diff_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	    # result_planes.append(diff_img)
	    result_norm_planes.append(norm_img)

	# result = cv2.merge(result_planes)
	result_norm = cv2.merge(result_norm_planes)
	return result_norm


if __name__ == '__main__':
	img = cv2.imread(sys.argv[1], -1)
	removed_shadow_img = remove_shadow_img(img)
	cv2.imwrite('shadows_out_norm.png', removed_shadow_img)
