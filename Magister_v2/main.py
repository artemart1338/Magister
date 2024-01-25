import os

import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import pandas as pd
from model import create_custom_model
from data_preprocessing import load_and_preprocess_images, split_data

def plot_history(history, size):
    plt.figure(figsize=(12, 5))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy for Size {size}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss for Size {size}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'plots/history_{size[0]}x{size[1]}.png')
    plt.close()

def main():
    with tf.device('/GPU:0'):  # Use GPU
        train_directory = "Test_png/train"
        test_directory = "Test_png/test"

        results = []
        sizes = [(128, 128), (256, 256), (512, 512)]

        for size in sizes:
            train_images, train_labels = load_and_preprocess_images(train_directory, target_size=size)
            (train_images, train_labels), (validation_images, validation_labels) = split_data(train_images, train_labels)
            test_images, test_labels = load_and_preprocess_images(test_directory, target_size=size)

            model = create_custom_model(size + (1,))
            history = model.fit(train_images, train_labels, epochs=10, batch_size=4, validation_data=(validation_images, validation_labels))
            test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
            print(f"\nAccuracy on test data for size {size}: {test_acc}")

            plot_history(history, size)
            results.append({'Size': f'{size[0]}x{size[1]}', 'Accuracy': test_acc})

            model.save_weights(f'weights/saved_model_weights_{size[0]}x{size[1]}.h5')

        df = pd.DataFrame(results)
        df.to_csv('results.csv', index=False)

if __name__ == "__main__":
    main()
