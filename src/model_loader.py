import tensorflow as tf
import keras
from .config import MODEL_PATH

def load_model():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model
#MODEL_PATH = "C:/Users/User/Desktop/emotion/models/resnet50_model2.keras"