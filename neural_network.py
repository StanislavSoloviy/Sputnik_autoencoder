import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import json
import main
import os
# from tensorflow.keras.applications import VGG16
from keras.layers import Input, Reshape, Conv2D, MaxPooling2D, UpSampling2D, InputLayer, Conv2DTranspose, Dense, Flatten, BatchNormalization, Dropout
from keras.models import Model


class NeuralNetwork:
    def __init__(self):
        self.__network_path = main.NETWORK_NAME
        self.__model = None
        self.__batch_size = main.BATCH_SIZE  # Размер батча
        self.__img_shape = main.IMG_SHAPE  # Разрешение
        self.__datebase_name = main.DATEBASE_NAME  # Относительный путь до базы данных
        self.__q_train = main.Q_TRAIN  # Количество обучающих примеров
        self.__epochs = main.EPOCHS  # Кол-во эпох


    def image2array(self, filelist, sp=False, SNR=0.5):
        """Преобразование изображений в векторы.
        sp- прогоняем изображенеи через фильтр соль/перец
        SNR- процент шума"""
        image_array = []
        for image in filelist:
            img = cv2.imread(image)
            if sp:
                img = self.add_salt_pepper(img.transpose(2, 1, 0), SNR)
                img = img.transpose(2, 1, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.__img_shape, self.__img_shape))
            image_array.append(img)
        image_array = np.array(image_array)
        image_array = image_array.reshape(image_array.shape[0], self.__img_shape, self.__img_shape, 3)
        image_array = image_array.astype('float32')
        image_array = image_array / 255.0
        return np.array(image_array)

    def __call__(self, *args, **kwargs):
        return self.__model(*args, **kwargs)

    def create(self):
        """Создание модели"""
        input_img = Input(shape=(self.__img_shape, self.__img_shape, 3))
        x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


        autoencoder = Model(input_img, decoded, name="autoencoder")
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.summary()
        self.__model = autoencoder

    def save(self):
        """Сохранение модели"""
        self.__model.save(self.__network_path)
        print("Модель сохранена")

    def load(self):
        """Загрузка модели"""
        try:
            self.__model = tf.keras.models.load_model(self.__network_path)
            print(f"Moдель {self.__network_path} загружена")
        except:
            print(f"Moдель {self.__network_path} не найдена")

    def train(self, sp=False, SNR=0.5):
        """Обучение модели
        sp=True - на вход моделиподаём изображения через фильтр соль/перец. На выходе ожидаем чистое изображение
        SNR- процент шума"""
        x_train = self.image2array([os.path.join(self.__datebase_name, filename)
                                    for filename in os.listdir(self.__datebase_name)][:self.__q_train], sp=sp, SNR=SNR)
        y_train = self.image2array([os.path.join(self.__datebase_name, filename)
                                    for filename in os.listdir(self.__datebase_name)][:self.__q_train])

        # Обучение модели
        self.__model.fit(x_train, y_train, batch_size=self.__batch_size,
                                   epochs=self.__epochs,
                                   validation_split=0.2)

        print("Модель обучена")
        self.save()

    def check(self, sp=False, SNR=0.5):
        """Проверка тренировочных данных
        sp=True - на вход модели подаём изображения через фильтр соль/перец
        SNR- процент шума"""
        n = 10
        x_train = self.image2array([os.path.join(self.__datebase_name, filename)
                                   for filename in os.listdir(self.__datebase_name)][self.__q_train:self.__q_train+n],
                                   sp=sp, SNR=SNR)
        decoded_imgs = self.__model.predict(x_train, batch_size=n)

        self.plot_digits(x_train, decoded_imgs)


    @staticmethod
    def add_salt_pepper(img, SNR):
        img_ = img.copy()
        c, h, w = img_.shape
        mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
        mask = np.repeat(mask, c, axis=0)  # Копировать по каналу в ту же форму, что и img
        img_[mask == 1] = 255  # солевой шум
        img_[mask == 2] = 0  # перцовый шум
        return img_

    @staticmethod
    def plot_digits(*args):
        args = [x.squeeze() for x in args]
        n = min([x.shape[0] for x in args])

        plt.figure(figsize=(2 * n, 2 * len(args)))
        for j in range(n):
            for i in range(len(args)):
                ax = plt.subplot(len(args), n, i * n + j + 1)
                plt.imshow(args[i][j])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.show()


