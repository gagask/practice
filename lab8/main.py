import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


class PA:
    def __init__(self):
        data = pd.read_csv('iris.data', header=None)
        self.X = data.iloc[:, :4].to_numpy()
        labels = data.iloc[:, 4].to_numpy()
        le = preprocessing.LabelEncoder()
        self.Y = le.fit_transform(labels)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.5)

    def lda(self):
        target_names = ['setosa', 'versicolor', 'virginica']
        y = self.y_train

        clf = LinearDiscriminantAnalysis()
        y_pred = clf.fit(self.X_train, self.y_train).predict(self.X_test)
        print("Количество наблюдений, которые были неправильно определены:", (self.y_test != y_pred).sum())
        print("Точность:", clf.score(self.X_train, self.y_train))

        X_r2 = clf.transform(self.X_train)
        plt.figure()
        colors = ['navy', 'turquoise', 'darkorange']
        lw = 2

        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color)

        plt.show()

    def plot_lda(self):
        f, ax = plt.subplots(1, 2)

        lin = np.linspace(0.05, 0.95, 19)

        arr1 = []
        arr2 = []
        for i in lin:
            gnb = LinearDiscriminantAnalysis(priors=[0.2, 0.7, 0.2])
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=i, random_state=830416)
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            arr1.append((y_test != y_pred).sum())
            arr2.append(gnb.score(X_test, y_test))
        ax[0].plot(lin, arr1)
        ax[1].plot(lin, arr2)

        ax[0].set_ylabel('Wrong classified')
        ax[1].set_ylabel('Score')

        plt.show()

    def svm(self):
        clf = svm.SVC()
        y_pred = clf.fit(self.X_train, self.y_train).predict(self.X_test)
        print("Количество наблюдений, которые были неправильно определены:", (self.y_test != y_pred).sum())
        print("Точность:", clf.score(self.X, self.Y))
        print(clf.support_vectors_)
        print(clf.support_)
        print(clf.n_support_)
        print("........")

    def plot_svm(self):
        f, ax = plt.subplots(1, 2)

        lin = np.linspace(0.05, 0.95, 19)

        arr1 = []
        arr2 = []
        for i in lin:
            gnb = svm.SVC()
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=i, random_state=830416)
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            arr1.append((y_test != y_pred).sum())
            arr2.append(gnb.score(X_test, y_test))
        ax[0].plot(lin, arr1)
        ax[1].plot(lin, arr2)

        ax[0].set_ylabel('Wrong classified')
        ax[1].set_ylabel('Score')

        plt.show()


def main():
    PA().plot_svm()


if __name__ == '__main__':
    main()
