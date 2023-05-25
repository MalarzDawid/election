import os
import face_recognition


class ElectionDataset:
    def __init__(self, dir_path: str) -> None:
        self.dir_path = dir_path
        self._loader = face_recognition.load_image_file
        self._encoder =  face_recognition.face_encodings
        self._known_face_names = []
        self._known_face_encodings = []

    def prepare(self, filename: str = "img1.png"):
        for item in os.listdir(self.dir_path):
            filepath = os.path.join(self.dir_path, item, filename)
            known_image = self._loader(filepath)
            encoding_image = self._encoder(known_image)[0]
            self._known_face_encodings.append(encoding_image)
            self._known_face_names.append(item)
    
    def get_names(self):
        return self._known_face_names
    
    def get_encodings(self):
        return self._known_face_encodings
