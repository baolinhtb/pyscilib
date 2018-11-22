import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import os, sys
from PIL import Image


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect


def get_approx_contours(cnts):
	# find the main island (biggest area)
	cnt = cnts[0]
	max_area = cv2.contourArea(cnt)
	cX,cY = 0,0
	for i,con in enumerate(cnts):
	    if cv2.contourArea(con) > max_area:
	        cnt = con
	        # cX, cY = get_center_contour(cnt)
	        max_area = cv2.contourArea(con)
	# define main island contour approx. and hull
	# perimeter = cv2.arcLength(cnt.copy(),True)
	epsilon = 0.01*cv2.arcLength(cnt.copy(),True)
	approx = cv2.approxPolyDP(cnt.copy(),epsilon,True)

	# cv2.circle(draw_ori_im, (cX, cY), 9, (155, 155, 0), 8)
	# cv2.drawContours(draw_ori_im, approx, -1, (122, 100, 255), 18)
	return approx


def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)

    return img.point(contrast)

def increase_contrast(im):
    pil_im = Image.fromarray(im)
    return np.array(change_contrast(pil_im, 100))


def perspective_warp_img(im):
	or_im = im.copy()
	# thres = cv2.adaptiveThreshold(im, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,31,28)
	im = increase_contrast(im)
	im = increase_contrast(im)
	im = increase_contrast(im)
	# thres = cv2.adaptiveThreshold(im, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,31,40)

	retval, thres = cv2.threshold(im, 120, 255, cv2.THRESH_BINARY)
	im2, cnts, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	approx = get_approx_contours(cnts)
	dst_point = np.float32([[0,0], [3508,0],[3508, 2480], [0,2480] ])
	coners_obj = order_points(np.float32([x[0] for x in approx.copy()]))
	PT = cv2.getPerspectiveTransform(coners_obj,dst_point)
	dst = cv2.warpPerspective(or_im.copy(),PT,(3508,2480))
	return dst

if __name__ == '__main__':
	folder_in = sys.argv[1]
	if not os.path.exists(folder_in):
		print('Input folder not exist!')
	folder_out = '{}_dewarp'.format(folder_in)
	if not os.path.exists(folder_out):
		os.mkdir(folder_out)
	image_list = os.listdir(folder_in)
	for f in image_list:
		if '_.' in f:
			continue
		im_path = os.path.join(folder_in, f)
		im = cv2.imread(im_path)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		out_im = perspective_warp_img(im.copy())

		cv2.imwrite(os.path.join(folder_out, f), out_im)



