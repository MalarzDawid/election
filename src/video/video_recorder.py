import cv2
from dataclasses import dataclass


@dataclass
class VideoRecorder:
    url: str
    output_dir: str = "data"

    def init(self):
        cap = cv2.VideoCapture(self.url)
        return cap
