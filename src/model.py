import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50
from keras import layers

def build_model(
    input_shape=(140, 140, 3),
    num_classes=7, base_learning_rate = 0.0001):

    """
    This function creates and compiles emotion classification model
    using ResNet-50 transfer learning.
    """
    #loading the pretrained ResNet-50 without the classifier head
    base_model = keras.applications.ResNet50(
    weights = 'imagenet',
    input_shape=(140, 140, 3),
    include_top= False)

    #freezing the entire base model first
    base_model.trainable = False

    #Create new classification head
    inputs = keras.Input(shape=(input_shape))
    #Pass inputs through ResNet backbone
    x = base_model(inputs, training = False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu') (x)
    x = layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(7, activation="softmax") (x)
    model = keras.Model(inputs, outputs)

    #Fine-tuning setup
    print("Number of layers in the base model are:", len(base_model.layers))
    fine_tune_from=100
    base_model.trainable = True
    #freezing the early layers
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    
    #Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=base_learning_rate / 10
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model




