import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import cv2  # Импортируем OpenCV

def load_and_preprocess_images(directory, target_size=(100, 100)):
    images = []
    labels = []

    # Перебор всех подпапок в директории
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        # Определение метки класса на основе имени папки
        label = 1 if 'forg' in folder_name else 0

        # Загрузка каждого изображения в папке
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, color_mode='grayscale', target_size=target_size)
            img_array = img_to_array(img)

            # Преобразование в полутоновое изображение и бинаризация
            # Перед бинаризацией не нормализуем изображение
            gray_image = np.squeeze(img_array.astype(np.uint8))  # Преобразование в 8-битное изображение
            thresh, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Добавляем бинарное изображение в массив
            images.append(np.expand_dims(binary_image, axis=-1))  # Возвращаем измерение канала
            labels.append(label)

    return np.array(images), np.array(labels)

def split_data(images, labels, train_size=0.8):
    # Разделение данных на обучающие и тестовые выборки
    num_train_samples = int(len(images) * train_size)
    return (images[:num_train_samples], labels[:num_train_samples]), (images[num_train_samples:], labels[num_train_samples:])

# Пример использования
if __name__ == "__main__":
    train_directory = "Test_png/train"
    test_directory = "Test_png/test"

    # Загрузка и предобработка обучающих изображений
    train_images, train_labels = load_and_preprocess_images(train_directory)

    # Загрузка и предобработка тестовых изображений
    test_images, test_labels = load_and_preprocess_images(test_directory)

    # Разделение на обучающие и тестовые наборы
    (train_images, train_labels), (validation_images, validation_labels) = split_data(train_images, train_labels)
