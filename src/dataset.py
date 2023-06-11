from db import VectorDB, DBIndexEnum
import numpy as np
from pathlib import Path
from deepface.basemodels import VGGFace
import cv2
import os


SIZE = 2622
DETECTOR_IMG_SIZE = (224, 224)
DATASET_DIR = Path("dataset/new")
MODEL_IMAGE_SIZE = (224, 224)


def create_dataset(
    dataset_dir: Path, db: VectorDB, model, detector, output: str = "db"
):
    for face_name in os.listdir(dataset_dir):
        for img in os.listdir(os.path.join(DATASET_DIR, face_name)):
            face_path = DATASET_DIR / face_name / img
            raw_image = cv2.imread(str(face_path))
            h, w, _ = raw_image.shape

            # Update input size to img size
            detector.setInputSize((w, h))

            # Detect all faces
            _, faces = detector.detect(raw_image)
            if faces is None:
                print(f"Error: could not detect faces in {face_path}")
                continue
            else:
                try:
                    x, y, w, h = [int(i) for i in faces[0][:4]]
                    detected_face = raw_image[y : y + h, x : x + w]
                except Exception as e:
                    print(f"Error: exception when processing faces in {face_path}: {e}")
                    detected_face = raw_image

                # Resize
                detected_face_resize = cv2.resize(detected_face, MODEL_IMAGE_SIZE)

                # Add batch
                detected_face_batch = np.expand_dims(detected_face_resize, axis=0)
                embedding = model.predict(detected_face_batch)
                db.append(face_name, embedding)

    db.save(str(output))


if __name__ == "__main__":
    # Load detector
    detector = cv2.FaceDetectorYN.create(
        "models/face_detection_yunet_2022mar.onnx", "", DETECTOR_IMG_SIZE
    )
    detector.setInputSize(DETECTOR_IMG_SIZE)

    # Load embedding model
    model = VGGFace.loadModel()

    # Db init
    db = VectorDB(DBIndexEnum.EUCLIDIAN, SIZE)

    create_dataset(DATASET_DIR, db, model, detector)
