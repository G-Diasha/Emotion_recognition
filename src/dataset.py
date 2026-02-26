import tensorflow as tf
from .config import BATCH_SIZE, IMAGE_SIZE, DATA_DIR
from keras.applications.resnet50 import preprocess_input

def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        color_mode="rgb",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        color_mode="rgb",
    )
    return train_ds, val_ds

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image

def prepare_dataset(ds):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds