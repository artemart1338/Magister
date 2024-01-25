import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
from model import create_model, compile_model
from keras.preprocessing.image import load_img, img_to_array

# Функция для предсказания модели
def model_predict(file_path, model):
    img = load_img(file_path, target_size=(100, 100), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    prediction = model.predict(np.array([img_array]))
    result = np.argmax(prediction, axis=1)
    return result

class SignatureVerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система идентификации рукописной подписи с помощью сверточных нейросетей")
        self.root.geometry("500x300")  # Установка размера окна
        self.root.resizable(True, True)
        # Загрузка обученной модели
        input_shape = (100, 100, 1)
        self.model = create_model(input_shape)
        self.model = compile_model(self.model)
        # TODO: Загрузите веса модели
        self.model.load_weights('weights/saved_model_weights.h5')

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):

        # Метка с инструкцией
        self.instruction_label = tk.Label(self.root,
                                          text="Чтобы проверить подпись на подлинность загрузите изображение",
                                          font=('Montserrat', 10))
        self.instruction_label.pack(pady=(10, 0))  # Располагаем над кнопкой

        # Создаем контейнер Frame для центрирования кнопки
        self.center_frame = tk.Frame(self.root)
        self.center_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, )

        # Кнопка для загрузки изображения внутри Frame
        self.load_button = tk.Button(self.center_frame, text="Загрузить изображение", font=('Montserrat', 10), command=self.load_image)
        self.load_button.pack()

        # Метка для вывода результата
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

    def load_image(self):
        # Загрузка изображения
        self.instruction_label.pack_forget()
        file_path = filedialog.askopenfilename()
        if file_path:  # Проверка, что путь файла существует
            self.display_image(file_path)  # Вызов функции для отображения изображения
            result = model_predict(file_path, self.model)  # Получение результата модели

            self.load_button.pack_forget()  # Сначала убираем кнопку
            self.load_button.pack(pady=(90, 0))  # Затем добавляем её обратно с отступом сверху

            # Отображение результата
            result_text = 'Подпись подлинная' if result[0] == 1 else 'Подпись поддельная'
            self.result_label.config(text=result_text,
                                     fg='dark green' if result[0] == 1 else 'dark red')


    def display_image(self, file_path):
        # Загрузка изображения с помощью Pillow
        img = Image.open(file_path)
        img.thumbnail((250, 250))  # Уменьшение размера если необходимо
        img = ImageTk.PhotoImage(img)

        # Если метка с изображением уже существует, обновляем изображение
        if hasattr(self, 'image_label'):
            self.image_label.configure(image=img)
            self.image_label.image = img
        else:
            # Создаем новую метку и отображаем изображение
            self.image_label = Label(self.root, image=img)
            self.image_label.image = img
            self.image_label.pack(before=self.result_label)


# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureVerificationApp(root)
    root.mainloop()
