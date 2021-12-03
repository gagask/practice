import pandas as pd
import numpy as np
from matplotlib import gridspec
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class DB:
    def __init__(self):
        self.data = pd.read_csv('CC GENERAL.csv').iloc[:,1:].dropna()
        self.no_labeled_data = self.data[1:]
        # print(data)
        self.data = np.array(self.data, dtype='float')
        min_max_scaler = preprocessing.StandardScaler()
        self.scaled_data = min_max_scaler.fit_transform(self.data)

    def dbscan_bas(self):
        clustering = DBSCAN().fit(self.scaled_data)
        print("Метки:", set(clustering.labels_))
        print("Количество кластеров:", len(set(clustering.labels_)) - 1)
        print("Процент не кластеризованных данных:", list(clustering.labels_).count(-1) / len(list(clustering.labels_)) * 100, "%")

    def plainDbKD(self):
        f, ax = plt.subplots(1, 2)

        arr1 = []
        arr2 = []
        lin = np.linspace(1.9, 2.1, 20)

        for i in lin:
            clustering = DBSCAN(eps=i, min_samples=3).fit(self.scaled_data)
            arr1.append(len(set(clustering.labels_)) - 1)
            arr2.append(list(clustering.labels_).count(-1) / len(list(clustering.labels_)) * 100)
        ax[0].plot(lin, arr1)
        ax[1].plot(lin, arr2)
        plt.show()

    def plainDbKK(self):
        f, ax = plt.subplots(1, 2)

        arr1 = []
        arr2 = []
        lin = np.linspace(5, 100, 20)

        for i in lin:
            clustering = DBSCAN(min_samples=i).fit(self.scaled_data)
            arr1.append(len(set(clustering.labels_)) - 1)
            arr2.append(list(clustering.labels_).count(-1) / len(list(clustering.labels_)) * 100)
        ax[0].plot(lin, arr1)
        ax[1].plot(lin, arr2)
        plt.show()

    def plainOpKD(self):
        f, ax = plt.subplots(1, 2)

        arr1 = []
        arr2 = []
        lin = np.linspace(1.8, 2.4, 20)

        for i in lin:
            clustering = OPTICS(max_eps=2, min_samples=3, cluster_method="dbscan").fit(self.scaled_data)
            arr1.append(len(set(clustering.labels_)) - 1)
            arr2.append(list(clustering.labels_).count(-1) / len(list(clustering.labels_)) * 100)
        ax[0].plot(lin, arr1)
        ax[1].plot(lin, arr2)
        plt.show()

    def plotRes(self):
        X = self.scaled_data
        db = DBSCAN(eps=2.05, min_samples=3).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        X = PCA(n_components=2).fit_transform(X)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # #############################################################################
        # Plot result

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.show()

    def plotOp(self):
        X = self.scaled_data

        clust = OPTICS(max_eps=2.05, min_samples=3, cluster_method='dbscan').fit(X)

        X = PCA(2).fit_transform(self.scaled_data)
        # Run the fit

        labels_050 = cluster_optics_dbscan(
            reachability=clust.reachability_,
            core_distances=clust.core_distances_,
            ordering=clust.ordering_,
            eps=0.5,
        )
        labels_200 = cluster_optics_dbscan(
            reachability=clust.reachability_,
            core_distances=clust.core_distances_,
            ordering=clust.ordering_,
            eps=2,
        )

        space = np.arange(len(X))
        reachability = clust.reachability_[clust.ordering_]
        labels = clust.labels_[clust.ordering_]

        plt.figure(figsize=(10, 7))
        G = gridspec.GridSpec(2, 3)
        ax1 = plt.subplot(G[0, :])
        ax2 = plt.subplot(G[1, 0])
        ax3 = plt.subplot(G[1, 1])
        ax4 = plt.subplot(G[1, 2])

        # Reachability plot
        colors = ["g.", "r.", "b.", "y.", "c."]
        for klass, color in zip(range(0, 5), colors):
            Xk = space[labels == klass]
            Rk = reachability[labels == klass]
            ax1.plot(Xk, Rk, color, alpha=0.3)
        ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
        ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
        ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
        ax1.set_ylabel("Reachability (epsilon distance)")
        ax1.set_title("Reachability Plot")

        # OPTICS
        colors = ["g.", "r.", "b.", "y.", "c."]
        for klass, color in zip(range(0, 5), colors):
            Xk = X[clust.labels_ == klass]
            ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
        ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], "k+", alpha=0.1)
        ax2.set_title("Automatic Clustering\nOPTICS")

        # DBSCAN at 0.5
        colors = ["g", "greenyellow", "olive", "r", "b", "c"]
        for klass, color in zip(range(0, 6), colors):
            Xk = X[labels_050 == klass]
            ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker=".")
        ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], "k+", alpha=0.1)
        ax3.set_title("Clustering at 0.5 epsilon cut\nDBSCAN")

        # DBSCAN at 2.
        colors = ["g.", "m.", "y.", "c."]
        for klass, color in zip(range(0, 4), colors):
            Xk = X[labels_200 == klass]
            ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
        ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], "k+", alpha=0.1)
        ax4.set_title("Clustering at 2.0 epsilon cut\nDBSCAN")

        plt.tight_layout()
        plt.show()


def main():
    DB().plotOp()


if __name__ == '__main__':
    main()