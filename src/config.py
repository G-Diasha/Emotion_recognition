import os
IMAGE_SIZE = (140, 140)
BATCH_SIZE = 64
CLASS_NAMES = [
    "angry","disgust","fear",
    "happy","neutral","sad","surprise"
]
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet50_model2.keras")
DATA_DIR = "c:\\Users\\User\\Desktop\\emotion\\dataset\\train"

