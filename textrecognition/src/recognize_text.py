import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import utils

from PIL import Image
import cv2

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

def get_label(model, images, use_lexicon, tree):
	converter = utils.strLabelConverter(alphabet)
	print "Shape: ", images.shape
	if torch.cuda.is_available():
	    images = images.cuda()
	images = Variable(images)

	model.eval()
	pred = model(images) # [26, b, 37]
	_, preds = pred.max(2)
	preds = preds.transpose(1, 0).contiguous().view(-1) # [b*26]
	preds_size = Variable(torch.IntTensor([pred.size(0)] * pred.size(1)))
	results = converter.decode_with_lexicon(pred.data, preds.data, preds_size.data, use_lexicon, tree)
	return results
