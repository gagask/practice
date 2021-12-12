import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
import matplotlib.pyplot as plt
from sklearn import tree



class BC:
    def __init__(self):
        data = pd.read_csv('iris.data', header=None)
        self.X = data.iloc[:, :4].to_numpy()
        # print("Data",X)
        labels = data.iloc[:, 4].to_numpy()
        # print("Lables",labels)
        le = preprocessing.LabelEncoder()
        self.Y = le.fit_transform(labels)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.5)
        # print("X_train",X_train)
        # print("X_test",X_test)
        # print("y_train",y_train)
        # print("y_test",y_test)

    def naive_b(self):
        gnb = GaussianNB()
        y_pred = gnb.fit(self.X_train, self.y_train).predict(self.X_test)
        # количество наблюдений, которые были неправильно определены
        print("Количество наблюдений, который были неправильно определены", (self.y_test != y_pred).sum())
        print("Точность классификации", gnb.score(self.X_train, self.y_train))

    def plot_NK(self):
        f, ax = plt.subplots(1, 2)

        lin = np.linspace(0.05, 0.95, 19)

        methods = [GaussianNB, MultinomialNB, ComplementNB, BernoulliNB ]

        m = methods[3]

        arr1 = []
        arr2 = []
        for i in lin:
            gnb = m()
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=i, random_state=830416)
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            arr1.append((y_test != y_pred).sum())
            arr2.append(gnb.score(X_test, y_test))
        ax[0].plot(lin, arr1)
        ax[1].plot(lin, arr2)
        plt.show()

    def tree_K(self):
        clf = tree.DecisionTreeClassifier()
        y_pred = clf.fit(self.X_train, self.y_train).predict(self.X_test)
        print("Количество наблюдений, который были неправильно определены", (self.y_test != y_pred).sum())
        print("Точность классификации", clf.score(self.X_train, self.y_train))
        print("Листьев:", clf.get_n_leaves(),"; Глубина:", clf.get_depth())

        plt.subplots(1, 1, figsize=(5, 5))
        tree.plot_tree(clf, filled=True)
        plt.show()

    def plot_TK(self):
        f, ax = plt.subplots(1, 2)

        lin = np.linspace(0.05, 0.95, 19)

        arr1 = []
        arr2 = []
        for i in lin:
            clf = tree.DecisionTreeClassifier()
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=i, random_state=830416)
            y_pred = clf.fit(X_train, y_train).predict(X_test)
            arr1.append((y_test != y_pred).sum())
            arr2.append(clf.score(X_test, y_test))
        ax[0].plot(lin, arr1)
        ax[1].plot(lin, arr2)
        plt.show()


def main():
    BC().plot_TK()


if __name__ == '__main__':
    main()
