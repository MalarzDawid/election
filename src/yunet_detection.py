import cv2


class YuNet:
    def __init__(
        self,
        model_path: str,
        input_size: list = [320, 320],
        conf_threshold: float = 0.9,
        nms_threshold: float = 0.3,
    ) -> None:
        self._model_path = model_path
        self._input_size = input_size
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold

        self._model = cv2.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._conf_threshold,
            nms_threshold=self._nms_threshold,
        )

    def set_input_size(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, frame):
        faces = self._model.detect(frame)
        return faces[1]
