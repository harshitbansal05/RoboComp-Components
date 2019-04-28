#
# Copyright (C) 2019 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

import sys, os, traceback, time
import torch
import cv2
import numpy as np
import time

import crnn
import detect_texts
from bktree import hamming_distance, BKTree

from PySide import QtGui, QtCore
from genericworker import *

NET_FILE = "assets/frozen_east_text_detection.pb"
MODEL_FILE = "assets/crnn.pth"
LEXICON_FILE = "assets/generic_lex.txt"

class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		self.timer.timeout.connect(self.compute)
		self.Period = 200
		self.timer.start(self.Period)

		# load the pre-trained EAST text detector
		print "[INFO] loading EAST text detector..."
		self.net = cv2.dnn.readNet(NET_FILE)
		self.model = crnn.CRNN(32, 1, 37, 256)
		if torch.cuda.is_available():
			self.model = self.model.cuda()
		print "[INFO] loading CRNN text recognizer..."
		self.model.load_state_dict(torch.load(MODEL_FILE))
		self.tree = None
		if self.use_lexicon:
			print "[INFO] loading generic english lexicon..."
			lexicon = []
			with open(LEXICON_FILE) as f:
				for line in f.read().splitlines():
					lexicon.append(line.lower())
			print "Length of the lexicon: ", len(lexicon)
			self.tree = BKTree(hamming_distance, lexicon)

	def setParams(self, params):
		return True

	@QtCore.Slot()
	def compute(self):
		print 'SpecificWorker.compute...'
		time1 = time.time()
		try:
			data = self.camerasimple_proxy.getImage()
			arr = np.fromstring(data.image, np.uint8)
			frame = np.reshape(arr, (data.width, data.height, data.depth))
			results = detect_texts.get_text(self.net, self.model, frame, self.use_lexicon, self.tree)
			texts = list()
			for result in results:
				box = result[0]
				mins = box.min(axis=0)
				xmin = mins[0]
				ymin = mins[1]
				maxs = box.max(axis=0)
				xmax = maxs[0]
				ymax = maxs[1]
				label = result[1]
				label = label + ","
				label = label + ""
				label += str(box[0][0]) + "," + str(box[0][1]) + "," + str(box[1][0]) + "," + str(box[1][1]) + \
				"," + str(box[2][0]) + "," + str(box[2][1]) + "," + str(box[3][0]) + "," + str(box[3][1]) 
				textData = SText()
				textData.startX = xmin
				textData.startY = ymin
				textData.endX = xmax
				textData.endY = ymax
				textData.label = label
				texts.append(textData)
			print "Time: ", time.time() - time1
			self.textList = texts
			
		except Ice.Exception, e:
			traceback.print_exc()
			print e

		cv2.waitKey(1)
		return True

	def getTextList(self):
		return self.textList
