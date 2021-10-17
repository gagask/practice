import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA,KernelPCA,SparsePCA, FactorAnalysis
import matplotlib.pyplot as plt


class My:
    def __init__(self):
        df = pd.read_csv('glass.csv')
        self.var_names = list(df.columns)  # получение имен признаков
        self.labels = df.to_numpy('int')[:, -1]  # метки классов
        self.data = df.to_numpy('float')[:, :-1]  # описательные признаки
        self.data = preprocessing.minmax_scale(self.data)

    def plot(self, data=None):
        if data is None:
            data = self.data

        fig, axs = plt.subplots(2, 4)
        for i in range(data.shape[1] - 1):
            axs[i // 4, i % 4].scatter(data[:, i], data[:, (i + 1)], c=self.labels, cmap='hsv')
            axs[i // 4, i % 4].set_xlabel(self.var_names[i])
            axs[i // 4, i % 4].set_ylabel(self.var_names[i + 1])
        plt.show()

    def forPCA(self):
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.data)
        print("Variance ratio:", pca.explained_variance_ratio_)
        print("Sum:", sum(pca.explained_variance_ratio_))
        print("Singular values:", pca.singular_values_)

        #self.plot(pca_data)

        #inv_data = pca.inverse_transform(pca_data)
        #self.plot(inv_data)

    def forKernelPCA(self):
        rezh = ["linear", "poly", "rbf", "sigmoid", "cosine"]
        fig, axs = plt.subplots(1, 5)
        for i in range(len(rezh)):
            data = KernelPCA(n_components=2, kernel=rezh[i]).fit_transform(self.data)
            axs[i].scatter(data[:, 0], data[:, 1], c=self.labels, cmap='hsv')
            axs[i].set_title(rezh[i])

        plt.show()


    def forSparcePCA(self):
        fig, axs = plt.subplots(2, 3)
        for i in range(0, 11, 2):
            data = SparsePCA(n_components=2, alpha=i/10).fit_transform(self.data)
            axs[i // 6, (i % 6)//2].scatter(data[:, 0], data[:, 1], c=self.labels, cmap='hsv')
            axs[i // 6, (i % 6)//2].set_title(f"alpha = {i/10}")

        plt.show()

    def forFact(self):
        pca = FactorAnalysis(n_components=2)
        data = pca.fit_transform(self.data)

        plt.scatter(data[:, 0], data[:, 1], c=self.labels, cmap='hsv')
        plt.show()

def main():
    My().forSparcePCA()


if __name__ == '__main__':
    main()
