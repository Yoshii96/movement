import numpy as np
import cv2
import argparse
from pathlib import Path


class MovementRecognition():

	def __init__(self, source_file, threshold, frames_to_analize):
		self.source_file = source_file
		self.threshold = threshold
		self.last_frames = None
		self.frames_to_analize = frames_to_analize

	def start(self):
		if self.source_file == 'camera':
			cap = cv2.VideoCapture(0)
		else:
			source = Path(self.source_file)
			if source.is_file():
				cap = cv2.VideoCapture(self.source_file)
			else:
				print('No such a file!')
				exit(1)

		while(True):
			ret, frame = cap.read()

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			if self.last_frames is None:
				self.last_frames = np.zeros(frame.shape)

			processed_frame = self.process_frame(frame)
			image_to_display = np.hstack((frame, processed_frame))
			try:
				cv2.imshow('frame', image_to_display)
			except TypeError:
				print('End of video')
				break

		cap.release()
		cv2.destroyAllWindows()

	def process_frame(self, frame):
		processed_frame = abs(frame - self.last_frames) > self.threshold
		self.last_frames = ((self.last_frames * (self.frames_to_analize -1)) + frame) / self.frames_to_analize
		
		processed_frame = processed_frame.astype(np.uint8)
		processed_frame[processed_frame == 1] = 255
		processed_frame[processed_frame == 0] = 0
		return processed_frame

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Movement recognition')
	parser.add_argument('--source_file',
						type=str,
						default='camera',
						help='Source file to analyze. If not given image from build-in camera')
	parser.add_argument('--threshold',
						type=int,
						choices=range(0,256),
						metavar="[0-255]",
						default=40,
						help='Change in pixel value recognized as movement')
	parser.add_argument('--frames_to_analize',
						type=int,
						default=10,
						choices=range(1,101),
						metavar="[0-100]",
						help='Number of frames that will be use to compare with threshold')
	args = parser.parse_args()
	movement_recognition = MovementRecognition(args.source_file, args.threshold, args.frames_to_analize)
	movement_recognition.start()