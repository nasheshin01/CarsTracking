import onnxruntime as rt
import cv2
import numpy as np
from numpy.linalg import norm


def read_image(path):
    orig_img = cv2.imread(path)
    img = cv2.resize(orig_img, (208, 208))
    img = img

    return img, orig_img


img, orig_img = read_image("7.png")
img2, orig_img2 = read_image("8.png")
providers = ['CPUExecutionProvider']
m = rt.InferenceSession("public\\vehicle-reid-0001\\osnet_ain_x1_0_vehicle_reid.onnx", providers=providers)
output_names = ["output"]
batch = np.array([img.astype(np.float32), img2.astype(np.float32)])
batch = np.swapaxes(batch, 1, 3)
batch = np.swapaxes(batch, 2, 3)
print(batch.shape)
onnx_pred = m.run(output_names, {"input": batch})

A = onnx_pred[0][0]
B = onnx_pred[0][1]
cosine = np.dot(A, B) / (norm(A) * norm(B))

# print('ONNX Predicted:', onnx_pred[0][0])
print('Cosine similarity:', cosine)