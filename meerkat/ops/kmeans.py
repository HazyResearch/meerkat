import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster

from meerkat import DataPanel, NumpyArrayColumn


def main():
    plot = False
    dp = DataPanel(
        {
            "age_of_gymnast": NumpyArrayColumn([30, 29, 20, 13, 12, 16, 30]),
            "ratings": NumpyArrayColumn([3.0, 3.1, 6.2, 9.2, 9.0, 9.1, 5.8]),
        }
    )

    # Groups here are (1, 3), (2, 6), (3, 9), (1, 9)

    X = np.hstack([dp["age_of_gymnast"][:, np.newaxis], dp["ratings"][:, np.newaxis]])
    # Number of clusters really matters.
    max_clusters = min(10, X.shape[0])
    ks = [k for k in range(1, max_clusters)]
    kmeans_algs = [cluster.KMeans(n_clusters=k).fit(X) for k in ks]
    sum_of_squares = np.square(X - X.mean(axis=0)).sum()
    alpha = 0.003
    kmeans_results = [(ks[i], alg.fit(X)) for i, alg in enumerate(kmeans_algs)]
    qualities = np.array([res.inertia_ / sum_of_squares for (_, res) in kmeans_results])
    penalties = np.array([alpha * k for (k, _) in kmeans_results])
    inertias = qualities + penalties
    best_k = np.argmin(inertias)
    print(inertias)
    print(f"Automatic ideal k: {ks[best_k]}")

    if plot:

        plt.scatter(
            dp["age_of_gymnast"], dp["ratings"], c=kmeans_results[best_k][1].labels_
        )
        plt.xlabel("Age of Gymnast")
        plt.ylabel("Rating")
        plt.title("Ratings for Gymnasts with respect to their ages.")
        plt.show()

    else:
        plt.plot(ks, penalties, label="Regularization")
        plt.plot(ks, qualities, label="Variance")
        plt.plot(ks, inertias, label="Inertia")
        plt.legend()
        plt.title(f"Inertia as a function of k at alpha = {alpha}")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.show()


if __name__ == "__main__":

    main()
