import tensorflow as tf
from tensorflow.python import keras
from keras import layers, models

def create_model(input_shape):
    model = models.Sequential()

    # Сверточный слой 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Сверточный слой 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Сверточный слой 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Вытягивание признаков в вектор
    model.add(layers.Flatten())

    # Полносвязный слой
    model.add(layers.Dense(64, activation='relu'))

    # Выходной слой с двумя выходами и softmax активацией
    model.add(layers.Dense(2, activation='softmax'))

    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
