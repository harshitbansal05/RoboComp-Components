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
import cv2
import numpy as np

from PySide import QtGui, QtCore
from genericworker import *

# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# sys.path.append('/opt/robocomp/lib')
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

class SpecificWorker(GenericWorker):
	def __init__(self, proxy_map):
		super(SpecificWorker, self).__init__(proxy_map)
		self.timer.timeout.connect(self.compute)
		self.Period = 200
		self.timer.start(self.Period)

	def setParams(self, params):
		return True

	@QtCore.Slot()
	def compute(self):
		print 'SpecificWorker.compute...'

		# Get image from camera
		data = self.camerasimple_proxy.getImage()
		arr = np.fromstring(data.image, np.uint8)
		frame = np.reshape(arr,(data.width, data.height, data.depth))

		# Get text list
		textL = self.textrecognition_proxy.getTextList()
		print textL

		# Showing data on the frame
		for textData in textL:
			startX = textData.startX
			startY = textData.startY
			endX = textData.endX
			endY = textData.endY
			box = textData.label.split(",")[1:]
			box = np.asarray(box, dtype=np.float32).reshape((4, 2))
			labels = textData.label.split(",")[0]
			labels = labels.split("&")
			if labels[1]:
				label = labels[0] + ", " + labels[1]
			else:
				label = labels[0]
			# cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
			cv2.polylines(frame, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 255), thickness=1)
			cv2.putText(frame, label, (startX, startY - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.imshow('Text', frame)

		return True
