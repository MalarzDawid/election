import cv2
import numpy as np

import face_recognition
from src.dataset import ElectionDataset


class FaceRecognition:
    def __init__(self, data_module: ElectionDataset) -> None:
        self.encodings = data_module.get_encodings()
        self.names = data_module.get_names()
        self._encoder = face_recognition.face_encodings
        self._compare = face_recognition.compare_faces
        self._distance = face_recognition.face_distance
        self.process_names = []

    def process(self, frame, bbox, vis: bool = False):
        unknown_encodings = self._encoder(frame, bbox)
        self.process_names = []
        for face_encoding in unknown_encodings:
            matches = self._compare(self.encodings, face_encoding)
            name = "Unknown"
            best_match_index = self.get_distance(face_encoding)
            if matches[best_match_index]:
                name = self.names[best_match_index]
            self.process_names.append(name)
        if vis:
            self.vis(frame, bbox)
        return self.process_names

    def vis(self, frame, unknown_faces):
        for (x, y, x1, y1), name in zip(unknown_faces, self.process_names):
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y1 - 35), (x1, y1), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y1 - 6), font, 1.0, (255, 255, 255), 1)

    def get_distance(self, unknown_encoding):
        face_distances = self._distance(self.encodings, unknown_encoding)
        best_match_index = np.argmin(face_distances)
        return best_match_index
