import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import main


class DataBase:
    """Класс для создания данных для обучения нейросети
    Данные представляют собой набор изображений, каждое изображение
    содержит от 2 до 7 случайных графиков со случайными параметрами
    """

    def __init__(self):
        self.__q_train = main.Q_TRAIN  # Количество обучающих примеров
        self.__q_test = main.Q_TEST  # Количество тестовых примеров
        self.__q_general = main.Q_GENERAL  # Общее количество примеров
        self.__datebase_name = main.DATEBASE_NAME  # Относительный путь до базы данных
        self.__x = main.X  # Диапазон входящих значений Х для функций

    # Метод, описывающая синусоиду
    @staticmethod
    def sin_f(a, w, f):
        def wrapper(x):
            return 10 * a * np.sin(w * x + f) + 100 * a

        return wrapper

    # Метод, описывающая параболу
    @staticmethod
    def quad(a, b, c):
        def wrapper(x):
            return a * x * x + b * x + 100 * c

        return wrapper

    # Метод, описывающий случайный выбор функции, и передающий ему случайные параметры
    def random_func(self):
        func = random.choice([self.sin_f, self.quad])
        #func = random.choice([self.quad])
        a, b, c = (random.randint(-5, 5) for _ in range(3))
        return func(a, b, c)

    # Создание единичного изображения
    def create_image(self, name, type="train"):
        plt.clf()
        x = self.__x
        quantity = random.randint(2, 7)
        for _ in range(quantity):
            func = self.random_func()
            y = np.array([func(i) for i in x])
            plt.plot(x, y, linewidth=3)
        path = os.path.join(self.__datebase_name, type, str(name) + '.png')
        plt.savefig(path)
        return {str(name) + '.png': quantity}

    # Очищение базы данных
    def clear_datebase(self, path):
        files = os.listdir(path)
        for f in files:
            p = os.path.join(path, f)
            if os.path.isdir(p):
                self.clear_datebase(p)
            else:
                os.remove(p)

    # Создание базы данных
    def create(self):
        self.clear_datebase(self.__datebase_name)
        annotation_dict = dict()
        for name in range(self.__q_train):
            annotation_dict.update(self.create_image(name, "train"))
        with open('datebase/annotations_train.json', 'w') as annotation_file:
            json.dump(annotation_dict, annotation_file, indent=3)

        annotation_dict = dict()
        for name in range(self.__q_train, self.__q_general):
            annotation_dict.update(self.create_image(name, "test"))
        with open('datebase/annotations_test.json', 'w') as annotation_file:
            json.dump(annotation_dict, annotation_file, indent=3)
        print("Создание базы данных завершено")
