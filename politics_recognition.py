import cv2
from src.yunet_detection import YuNet
from src.video.video_recorder import VideoRecorder
from src.utils import bbox_cut
from src.face_recognition import FaceRecognition
from src.dataset import ElectionDataset


SOURCE = "https://n-22-6.dcs.redcdn.pl/livehls/o2/sejm/ENC01/live.livx/playlist.m3u8?startTime=706690800000&stopTime=706742496000"
SOURCE = "https://sdt-epix7-54.tvp.pl/token/video/live/69672433/20230526/3114077478/VzQ4R0kVIYo9Jz5FYQw-vstNdeZPi40LwsKyADEh_5fi6Snk4cweHatdhkJNEgIwLruZ55sl3PKeKyyDXkw68rP6sUveBzxdk5rLkaU8BVvnmc8iyXmGjYQjFC4SzAaa3SvnzOGPtvA_KiX8bIWHYHsvlaJL4YOmdwxKSbwY_BA/master.m3u8"

def politics_recognition():
    # Init VideoRecorder
    recorder = VideoRecorder(SOURCE)
    cap = recorder.init()
 
    # Init face detector
    detector = YuNet("models/face_detection_yunet_2022mar.onnx")
    
    # Prepare dataset
    ds = ElectionDataset("output")
    ds.prepare()

    # Init face recognition
    fr = FaceRecognition(ds)
    
    # main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape

        # Detection part
        detector.set_input_size([w, h])
        dets = detector.infer(frame)
        
        # Recognition part
        unknow_faces = bbox_cut(dets)
        fr.process(frame, unknow_faces, vis=True)
        
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    politics_recognition()
        
       