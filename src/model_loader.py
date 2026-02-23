import tensorflow as tf
import keras
from .config import MODEL_PATH

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
