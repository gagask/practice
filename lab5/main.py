import time
import random
import math
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class KM:
    def __init__(self):
        self.data = pd.read_csv('iris.data', header=None)
        self.no_labeled_data = self.data[self.data.columns.drop(4)]

        # gen random data
        data1 = np.zeros([250, 2])
        for i in range(250):
            r = random.uniform(1, 3)
            a = random.uniform(0, 2 * math.pi)
            data1[i, 0] = r * math.sin(a)
            data1[i, 1] = r * math.cos(a)

        data2 = np.zeros([500, 2])
        for i in range(500):
            r = random.uniform(5, 9)
            a = random.uniform(0, 2 * math.pi)
            data2[i, 0] = r * math.sin(a)
            data2[i, 1] = r * math.cos(a)

        self.rand_data = np.vstack((data1, data2))

    def kMeans(self):
        k_means = KMeans(init='k-means++', n_clusters=3, n_init=15)
        k_means.fit(self.no_labeled_data)
        k_means_cluster_centers = k_means.cluster_centers_
        k_means_labels = pairwise_distances_argmin(self.no_labeled_data, k_means_cluster_centers)

        f, ax = plt.subplots(1, 3)
        colors = ['#4EACC5', '#FF9C34', '#4E9A06']
        print(ax)
        for i in range(3):
            my_members = k_means_labels == i
            cluster_center = k_means_cluster_centers[i]
            for j in range(3):
                ax[j].plot(self.no_labeled_data[my_members][j],
                           self.no_labeled_data[my_members][j + 1], 'w',
                           markerfacecolor=colors[i], marker='o', markersize=4)
                ax[j].plot(cluster_center[j], cluster_center[j + 1], 'o',
                           markerfacecolor=colors[i],
                           markeredgecolor='k', markersize=8)
        plt.show()

    def pcadec(self):
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.no_labeled_data)
        kmeans = KMeans(init='k-means++', n_clusters=3, n_init=15)
        kmeans.fit(reduced_data)
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
        )

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )
        plt.title(
            "K-means clustering on the digits dataset (PCA-reduced data)\n"
            "Centroids are marked with white cross"
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def elbowPlot(self):
        wcss = []
        for i in range(1, 15):
            kmean = KMeans(n_clusters=i, init="k-means++")
            kmean.fit_predict(self.no_labeled_data)
            wcss.append(kmean.inertia_)

        plt.plot(range(1, 15), wcss)
        plt.title('The Elbow Method')
        plt.xlabel("No of Clusters")
        plt.ylabel("WCSS")
        plt.show()

    def compKM(self):
        batch_size = 45
        n_clusters = 3

        # pca = PCA(n_components=2)
        # X = pca.fit_transform(self.no_labeled_data)

        # print(X)

        X = self.no_labeled_data

        # #############################################################################
        # Compute clustering with Means

        k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
        t0 = time.time()
        k_means.fit(X)
        t_batch = time.time() - t0

        # #############################################################################
        # Compute clustering with MiniBatchKMeans

        mbk = MiniBatchKMeans(
            init="k-means++",
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init=10,
            max_no_improvement=10,
            verbose=0,
        )
        t0 = time.time()
        mbk.fit(X)
        t_mini_batch = time.time() - t0

        # #############################################################################
        # Plot result

        fig = plt.figure(figsize=(8, 3))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

        # We want to have the same colors for the same cluster from the
        # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
        # closest one.
        k_means_cluster_centers = k_means.cluster_centers_
        order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
        mbk_means_cluster_centers = mbk.cluster_centers_[order]

        k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
        mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

        # KMeans
        ax = fig.add_subplot(1, 3, 1)
        for k, col in zip(range(n_clusters), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            ax.plot(X[my_members][0], X[my_members][1], "w", markerfacecolor=col, marker=".")
            ax.plot(
                cluster_center[0],
                cluster_center[1],
                "o",
                markerfacecolor=col,
                markeredgecolor="k",
                markersize=6,
            )
        ax.set_title("KMeans")
        ax.set_xticks(())
        ax.set_yticks(())
        plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_batch, k_means.inertia_))

        # MiniBatchKMeans
        ax = fig.add_subplot(1, 3, 2)
        for k, col in zip(range(n_clusters), colors):
            my_members = mbk_means_labels == k
            cluster_center = mbk_means_cluster_centers[k]
            ax.plot(X[my_members][0], X[my_members][1], "w", markerfacecolor=col, marker=".")
            ax.plot(
                cluster_center[0],
                cluster_center[1],
                "o",
                markerfacecolor=col,
                markeredgecolor="k",
                markersize=6,
            )
        ax.set_title("MiniBatchKMeans")
        ax.set_xticks(())
        ax.set_yticks(())
        plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_mini_batch, mbk.inertia_))

        # Initialise the different array to all False
        different = mbk_means_labels == 4
        ax = fig.add_subplot(1, 3, 3)

        for k in range(n_clusters):
            different += (k_means_labels == k) != (mbk_means_labels == k)

        identic = np.logical_not(different)
        ax.plot(X[identic][0], X[identic][1], "w", markerfacecolor="#bbbbbb", marker=".")
        ax.plot(X[different][0], X[different][1], "w", markerfacecolor="m", marker=".")
        ax.set_title("Difference")
        ax.set_xticks(())
        ax.set_yticks(())

        plt.show()

    def HK(self, n_clusters=3):
        hier = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        hier = hier.fit(self.no_labeled_data)
        hier_labels = hier.labels_

        f, ax = plt.subplots(1, 3)
        colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FF0000', '#000080']
        for i in range(n_clusters):
            my_members = hier_labels == i
            for j in range(3):
                ax[j].plot(self.no_labeled_data[my_members][j],
                           self.no_labeled_data[my_members][j + 1], 'w',
                           markerfacecolor=colors[i], marker='o', markersize=4)
        plt.show()

    def dendrogram(self):
        linked = linkage(self.no_labeled_data)
        dendrogram(linked, truncate_mode='level', p=6)
        plt.show()

    def randomData(self, linkage='ward'):
        hier = AgglomerativeClustering(n_clusters=2, linkage=linkage)
        hier = hier.fit(self.rand_data)
        hier_labels = hier.labels_

        my_members = hier_labels == 0
        plt.plot(self.rand_data[my_members, 0], self.rand_data[my_members, 1], marker='o',
                 markersize=4,
                 color='red', linestyle='None')
        my_members = hier_labels == 1
        plt.plot(self.rand_data[my_members, 0], self.rand_data[my_members, 1], marker='o',
                 markersize=4,
                 color='blue', linestyle='None')
        plt.show()


def main():
    KM().randomData('single')


if __name__ == '__main__':
    main()
