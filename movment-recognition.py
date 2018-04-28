import numpy as np
import cv2
import argparse
from pathlib import Path


class MovmentRecognition():

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
                self.last_frames = np.zeros((frame.shape[0], frame.shape[1]))

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
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #z obrazu na szary
        
        tmp = abs(grey - self.last_frames) > self.threshold
        self.last_frames = (self.last_frames * (self.frames_to_analize -1) + grey) / self.frames_to_analize
        
        vfunc = np.vectorize(self.map_to_255)
        grey = vfunc(tmp)
        
        #z szarego na obraz
        grey = np.reshape(grey, (grey.shape[0], grey.shape[1], 1),)
        final_frame = np.zeros((grey.shape[0],grey.shape[1],3), dtype=np.uint8)
        final_frame[:, :, :-2] = grey
        return final_frame

    def map_to_255(self, value):
        if value == False:
            return 0
        else:
            return 255

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Movment recognition')
	parser.add_argument('--source_file',
						type=str,
						default='camera',
						help='Source file to analyze. If not given image from build-in camera')
	parser.add_argument('--threshold',
						type=int,
						default=30,
						help='Change in pixel value recognized as movment')
	parser.add_argument('--frames_to_analize',
						type=int,
						default=10,
						help='Number of frames that will be use to compare with threshold')
	args = parser.parse_args()
	movment_recognition = MovmentRecognition(args.source_file, args.threshold, args.frames_to_analize)
	movment_recognition.start()