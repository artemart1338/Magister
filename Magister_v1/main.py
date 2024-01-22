import os
import tensorflow as tf
import logging

# Отключение предупреждений oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Отключение предупреждений TensorFlow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Отключение предупреждений о CPU features
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from model import create_model, compile_model
from data_preprocessing import load_and_preprocess_images, split_data


def main():
    # Пути к папкам с обучающими и тестовыми данными
    train_directory = "Test_png/train"
    test_directory = "Test_png/test"

    # Загрузка и предобработка обучающих изображений
    train_images, train_labels = load_and_preprocess_images(train_directory)

    # Разделение на обучающие и валидационные выборки
    (train_images, train_labels), (validation_images, validation_labels) = split_data(train_images, train_labels)

    # Загрузка и предобработка тестовых изображений
    test_images, test_labels = load_and_preprocess_images(test_directory)

    # Создание и компиляция модели
    input_shape = (100, 100, 1)  # Для черно-белых изображений размером 100x100
    model = create_model(input_shape)
    model = compile_model(model)

    # Обучение модели
    model.fit(train_images, train_labels, epochs=10, batch_size=4, validation_data=(validation_images, validation_labels))

    # Тестирование модели
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nТочность на тестовых данных: {test_acc}")

# Сохранение весов модели
    model.save_weights('weights/saved_model_weights.h5')

if __name__ == "__main__":
    main()

