import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
import matplotlib.pyplot as plt
from sklearn import tree, svm


class ZOO:
    def __init__(self):
        self.data = pd.read_csv('zoo.data', header=None)

        # Берем все признаки
        self.X = self.data.iloc[:, 1:17].to_numpy()
        # Берем столбец с информацией о принадлежности конкретному классу
        labels = self.data.iloc[:, 17].to_numpy()
        # Приводим к нормальному виду
        le = preprocessing.LabelEncoder()
        self.Y = le.fit_transform(labels)
        # Создаем обучающий и тестирующий наборы данных
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.3)

    def compare(self, times=100):
        """
        Метод сравнения моделей через среднее значение точности при times итерациях
        """
        # Список методов
        names = ["GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB", "LinearDiscriminantAnalysis",
                 "DecisionTreeClassifier", "SVC"]
        methods = [GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, LinearDiscriminantAnalysis,
                   tree.DecisionTreeClassifier, svm.SVC]

        methodsNC = [0] * len(methods)
        methodsAc = [0] * len(methods)

        for _ in range(times):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.3)
            for i in range(len(methods)):
                gnb = methods[i]()
                y_pred = gnb.fit(self.X_train, self.y_train).predict(self.X_test)

                methodsNC[i] += (self.y_test != y_pred).sum()
                methodsAc[i] += gnb.score(self.X_test, self.y_test)
        for i in range(len(methods)):
            print("Для ", names[i], "\tНе классифицировано", methodsNC[i]/times, "\tТочность:", methodsAc[i]/times)

    def accuracy(self, method=GaussianNB):
        """
        Метод выводит информацию о точности классификации и сравнивает с тестовыми данными
        """
        gnb = method()
        y_pred = gnb.fit(self.X_train, self.y_train).predict(self.X_test)

        print("Не классифицировано", (self.y_test != y_pred).sum(), "\tТочность:", gnb.score(self.X_test, self.y_test))

        print("Результат классификатора:\t", y_pred)
        print("Тестовые данные:\t\t\t", self.y_test)

    def accuracy_plot(self, method=GaussianNB):
        """
        График изменения точности классификации от размера данных обучения
        """
        f, ax = plt.subplots(1, 2)

        lin = np.linspace(0.05, 0.95, 19)

        arr1 = [0] * 19
        arr2 = [0] * 19
        for _ in range(10):
            for i in range(19):
                gnb = method()
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=lin[i])
                y_pred = gnb.fit(X_train, y_train).predict(X_test)
                arr1[i] += (y_test != y_pred).sum()
                arr2[i] += gnb.score(X_test, y_test)
        ax[0].plot(lin, list(map(lambda x: x/10, arr1)))
        ax[1].plot(lin, list(map(lambda x: x/10, arr2)))

        ax[0].set_ylabel('Wrong classified')
        ax[1].set_ylabel('Score')

        plt.show()

    def static_analysis(self):
        """
        Метод для анализа данных, лежащих в наборе данных
        """
        d = ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed",
             "backbone", "breathes", "venomous", "fins", "tail", "domestic", "catsize"]

        data = self.data.iloc[:, 1:].to_numpy()
        stats = []

        for cat in range(1, 8):
            cur = list(filter(lambda x: x[-1] == cat, data))
            stats.append([])
            for pr in range(16):
                tmp = list(filter(lambda x: x[pr] == 1, cur))
                if pr == 12:
                    continue
                stats[cat - 1].append(round(100 * len(tmp) / len(cur), 1))

        stats = pd.DataFrame(np.array(stats).T)
        stats.index = d
        print(stats)


def main():
    ZOO().accuracy()


main()
