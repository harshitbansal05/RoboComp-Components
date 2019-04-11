import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import utils

from PIL import Image
import cv2

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

def get_label(model, img, use_lexicon, tree):
	converter = utils.strLabelConverter(alphabet)

	transformer = resizeNormalize((100, 32))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(img).convert('L')
	image = transformer(image)
	if torch.cuda.is_available():
	    image = image.cuda()
	image = image.view(1, *image.size())
	image = Variable(image)

	model.eval()
	pred = model(image)
	_, preds = pred.max(2)
	preds = preds.transpose(1, 0).contiguous().view(-1)

	preds_size = Variable(torch.IntTensor([preds.size(0)]))
	a, b = converter.decode_with_lexicon(pred.data, preds.data, preds_size.data, use_lexicon, tree)
	return (a, b)
