import onnxruntime as rt
import cv2
import numpy as np
from numpy.linalg import norm


class VehicleIdentificator:

    def __init__(self, model_path) -> None:
        self.model = rt.InferenceSession(model_path)

    def get_embeddings(self, images):
        images_to_process = []
        for image in images:
            images_to_process.append(cv2.resize(image, (208, 208)).astype(np.float32))

        batch = np.array(images_to_process)
        batch = np.swapaxes(batch, 1, 3)
        batch = np.swapaxes(batch, 2, 3)

        onnx_pred = self.model.run(["output"], {"input": batch})

        return onnx_pred[0]

    def compare(self, emb1, emb2):
        return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))