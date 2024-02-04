import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import dlib
from PIL import Image

MODEL = tf.keras.models.load_model('engageplus.h5')


def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
  rects = detector(image, 1)

  w_max = 0
  h_max = 0

  found_face = False
  for faceRect in rects:
    rect = faceRect.rect
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    if (w*h)>(w_max*h_max):
      w_max=w
      h_max=h
      found_face = True
      img_crop = image[abs(rect.top()):abs(rect.bottom()), abs(rect.left()):abs(rect.right())]

  if found_face:
      img_crop = Image.fromarray(np.uint8(img_crop))
      resized_crop = img_crop.resize((256, 256), Image.LANCZOS)
      data_crop = np.asarray(resized_crop, dtype="uint8").reshape([256,256])
  else:
      return None

  data_crop = data_crop - np.mean(data_crop)
  f_b = np.sqrt(np.sum(np.square(data_crop)))

  if f_b==0:
    return None

  data_crop = data_crop * (100 / f_b)

  return data_crop

def predict(img):
    d = {0: 'Engaged', 1: 'Not Engaged'}
    img = format_image(img)
    img = np.stack((img,)*3, axis=-1)
    img = np.expand_dims(img, axis=0)
    pred = MODEL.predict(img)
    win = np.argmax(pred)
    pred_class = d[win]
    pred_conf = pred[0][win]
    return pred_class, pred_conf