import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from constants.constants import ROOT_DIR
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod


class Dataframe:
    def __init__(self):
        self.DATASET_PATH = ROOT_DIR / 'media' / 'sports_articles.tsv'

    def get(self):
        return pd.read_csv(self.DATASET_PATH, sep='\t')


class Reducer:
    def pca_reduce(self, X: spmatrix):
        reducer = PCA(n_components=2)
        return reducer.fit_transform(X)

    def tsne_reduce(self, X: spmatrix):
        reducer = TSNE(
            n_components=2,          # 2D for visualization
            perplexity=14,           # Lower value to focus on local structure
            learning_rate=100,       # Higher learning rate for cluster exaggeration
            early_exaggeration=10,   # Increase early exaggeration
            n_iter=500,            # More iterations for better convergence
            random_state=42,
        )
        return reducer.fit_transform(X)


class Vectorizer:
    def tfidf_vectorize(self, raw_documents: list[str]):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(raw_documents)

    def count_vectorizer(self, raw_documents: list[str]):
        vectorizer = CountVectorizer()
        vectorized_doc = vectorizer.fit_transform(raw_documents)

        if not isinstance(vectorized_doc, spmatrix):
            raise Exception(
                '[count_vectorizer] vectorized document id not instance of spmatrix.')

        return vectorized_doc


class Clustering(ABC):
    @abstractmethod
    def _dim_reduce(self, X: spmatrix) -> NDArray[np.float64]:
        """Сжатие матрицы документов. Можно использовать разные методы сжатия
        """
        pass

    @abstractmethod
    def _vectorize(self, raw_documents: list[str]) -> spmatrix:
        """Векторизатор для множества документов
        """
        pass

    @abstractmethod
    def plot_show(self, *args, **kwargs):
        """Отобразить кластеры на графике
        """
        pass

    @abstractmethod
    def cluterize(self) -> NDArray[np.float64]:
        pass


class KMeanClustering(Clustering):
    def __init__(self, documents: list[str]):
        self.__CLUSTERS_AMOUNT = 3
        self.__RAND_STATE = 42

        self.__X_reduced = np.array([])
        self.__documents = documents
        self.__labels = np.array([])

    def _dim_reduce(self, X: spmatrix) -> NDArray[np.float64]:
        reducer = Reducer()
        return reducer.pca_reduce(X)

    def _vectorize(self, raw_documents: list[str]) -> spmatrix:
        vectorizer = Vectorizer()
        return vectorizer.count_vectorizer(raw_documents)

    def plot_show(self):
        plt.figure(figsize=(12, 8))

        for cluster in range(self.__CLUSTERS_AMOUNT):
            plt.scatter(
                self.__X_reduced[self.__labels == cluster, 0],
                self.__X_reduced[self.__labels == cluster, 1],
                label=f'Cluster {cluster + 1}',
                alpha=0.7
            )

        plt.title('2D visualization of news articles clusters')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.show()

    def cluterize(self):
        X = self._vectorize(self.__documents)

        kmeans = KMeans(n_clusters=self.__CLUSTERS_AMOUNT,
                        random_state=self.__RAND_STATE)
        self.__labels = kmeans.fit_predict(X)

        self.__X_reduced = self._dim_reduce(X)

        if not isinstance(self.__X_reduced, np.ndarray):
            raise TypeError(
                "[cluterize] The method must return an ndarray array with float elements.")

        return self.__X_reduced


class SimpleClustering(Clustering):
    def __init__(self, documents: list[str]):
        self.__documents = documents
        self.__X_reduced = np.array([])

    def _dim_reduce(self, X: spmatrix) -> NDArray[np.float64]:
        reducer = Reducer()
        return reducer.pca_reduce(X)

    def _vectorize(self, raw_documents: list[str]):
        vectorizer = Vectorizer()
        return vectorizer.tfidf_vectorize(raw_documents)

    def plot_show(self):
        plt.figure(figsize=(14, 8))
        plt.scatter(self.__X_reduced[:, 0], self.__X_reduced[:, 1], alpha=0.6)
        plt.title('Simple Clustering')

        for i, _ in enumerate(self.__documents):
            plt.annotate(
                f'{i}', (self.__X_reduced[i, 0], self.__X_reduced[i, 1]), fontsize=6)

        plt.grid(True)
        plt.show()

    def cluterize(self) -> NDArray[np.float64]:
        X = self._vectorize(self.__documents)
        self.__X_reduced = self._dim_reduce(X)

        print(self.__X_reduced)
        if not isinstance(self.__X_reduced, np.ndarray):
            raise TypeError(
                "[cluterize] The method must return an ndarray array with float elements.")

        return self.__X_reduced


def main():
    dataframe = Dataframe()
    df = dataframe.get()

    # define vectorizer
    documents = df['summary'].to_list()

    # Simple clustarization
    # simple_clustering = SimpleClustering(documents=documents)
    # clusters = simple_clustering.cluterize()
    # print(clusters)
    # simple_clustering.plot_show()

    # Cluster using k-means
    kmeans_clustering = KMeanClustering(documents=documents)
    clusters = kmeans_clustering.cluterize()
    print(clusters)
    kmeans_clustering.plot_show()


if __name__ == "__main__":
    main()
