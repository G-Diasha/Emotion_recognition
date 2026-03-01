import numpy as np
import tensorflow as tf
import keras
import cv2
from keras.preprocessing.image import load_img
from .config import CLASS_NAMES, IMAGE_SIZE
from src.model_loader import load_model
from src.dataset import preprocess
from keras.applications.resnet50 import preprocess_input

file_path = "C:/Users/User/Desktop/emotion/dataset/train/surprise/Training_348814.jpg"

def predict_emotion(pic):
    #img= cv2.imread(pic)
    img= cv2.resize(pic,(IMAGE_SIZE))
    img = np.array(img)
    image = preprocess(img)
    #print("Shape:", image.shape)
    model = load_model()
    preds = model.predict(image)
    idx = np.argmax(preds, axis=1)[0]
    result = CLASS_NAMES[idx]
    return result


#img = load_img("C:/Users/User/Desktop/emotion/dataset/train/happy/Training_80015.jpg")

#img.show()


