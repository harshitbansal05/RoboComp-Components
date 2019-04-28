import numpy as np
import cv2
import time

import torch
from PIL import Image
import torchvision.transforms as transforms

import locality_aware_nms as nms_locality
from icdar import restore_rectangle

import recognize_text


class resizeNormalize(object):
	def __init__(self):
		self.toTensor = transforms.ToTensor()

	def __call__(self, img):
		img = self.toTensor(img)
		img.sub_(0.5).div_(0.5)
		return img


def resize_image(im, max_side_len=480):
	h, w, _ = im.shape

	resize_w = w
	resize_h = h

	if max(resize_h, resize_w) > max_side_len:
		ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
	else:
		ratio = 1.
	resize_h = int(resize_h * ratio)
	resize_w = int(resize_w * ratio)

	resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
	resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
	resize_h = max(32, resize_h)
	resize_w = max(32, resize_w)
	im = cv2.resize(im, (int(resize_w), int(resize_h)))

	ratio_h = resize_h / float(h)
	ratio_w = resize_w / float(w)

	return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
	if len(score_map.shape) == 4:
		score_map = score_map[0, :, :, 0]
		geo_map = geo_map[0, :, :, ]
	
	xy_text = np.argwhere(score_map > score_map_thresh)
	
	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	
	start = time.time()
	text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
	print('{} text boxes before nms'.format(text_box_restored.shape[0]))
	boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = text_box_restored.reshape((-1, 8))
	boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
	timer['restore'] = time.time() - start
	
	start = time.time()
	boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
	
	timer['nms'] = time.time() - start

	if boxes.shape[0] == 0:
		return None, timer

	for i, box in enumerate(boxes):
		mask = np.zeros_like(score_map, dtype=np.uint8)
		cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
		boxes[i, 8] = cv2.mean(score_map, mask)[0]
	boxes = boxes[boxes[:, 8] > box_thresh]

	return boxes, timer


def sort_poly(p):
	min_axis = np.argmin(np.sum(p, axis=1))
	p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
	if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
		return p
	else:
		return p[[0, 3, 2, 1]]


def get_text(net, model, image_orig, use_lexicon, tree):
	image, (ratio_h, ratio_w) = resize_image(image_orig)
	(H, W) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]
	
	timer = {'net': 0, 'restore': 0, 'nms': 0}
	start = time.time()

	(scores, geometry) = net.forward(layerNames)
	score = np.transpose(scores, (0, 2, 3, 1))
	geometry = np.transpose(geometry, (0, 2, 3, 1))
	timer['net'] = time.time() - start

	boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
	if boxes is None:
		return []

	boxes = boxes[:, :8].reshape((-1, 4, 2))
	boxes[:, :, 0] /= ratio_w
	boxes[:, :, 1] /= ratio_h
	
	num = 0
	for i, box in enumerate(boxes):
		box = sort_poly(box.astype(np.int32))
		boxes[i] = box
		if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
			continue
		num = num + 1
	if num == 0:
		return []

	rois = torch.zeros((num, 1, 32, 100))
	coords = []

	for i, box in enumerate(boxes):
		dst = np.asarray([[3, 3], [97, 3], [97, 28], [3, 28]])
		trans = cv2.getPerspectiveTransform(np.float32(box), np.float32(dst))
		rotated = cv2.warpPerspective(image_orig, trans, (100, 32), flags=cv2.INTER_LINEAR)
		cv2.imwrite('rotated{index}.png'.format(index=i), rotated)
		roi = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
		roi = Image.fromarray(roi).convert('L')
		roi = roi.resize((100, 32), Image.BILINEAR)
		transformer = resizeNormalize()
		roi = transformer(roi)
		rois[i] = roi
		coords.append(box)

	results = recognize_text.get_label(model, rois, use_lexicon, tree)
	print(results)
	print(coords)
	for i, coord in enumerate(coords):
		results[i] = (coord, results[i]) 
	print(results)
	return results
