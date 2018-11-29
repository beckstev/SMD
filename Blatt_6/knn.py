import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class KNN:
    '''KNN Classifier.

    Attributes
    ----------
    k : int
        Number of neighbors to consider.
    '''
    def __init__(self, k):
        '''Initialization.
        Parameters are stored as member variables/attributes.

        Parameters
        ----------
        k : int
            Number of neighbors to consider.
        '''
        self.k = k

        # Traning dataset
        self.traning_dataset = None

        # Predicted label of the test traning_set
        self.label_traning_dataset = None

        # Predicted label of the real dataset
        self.label_predict = None


    def fit(self, X, y):
        '''Fit routine.
        Training data is stored within object.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_attributes)
            Training data.
        y : numpy.array shape=(n_samples)
            Training labels.
        '''
        # Code
        self.traning_dataset = X
        self.label_traning_dataset = y


    def predict(self, X_pred):
        '''Prediction routine.
        Predict class association of each sample of X.

        Parameters
        ----------
        X : numpy.array, shape=(n_samples, n_attributes)
            Data to classify.

        Returns
        -------
        prediction : numpy.array, shape=(n_samples)
            Predictions, containing the predicted label of each sample.
        '''
        # Code
        self.label_predict = np.zeros(len(X_pred))

        dataset = self.traning_dataset
        labelset = self.label_traning_dataset

        for i in range(X_pred.shape[0]):
            # calculate the distance between every point and the test point
            # print(X_pred[i].reshape(1, X_pred[i].shape[0]), dataset)
            distance = cdist(X_pred[i].reshape(1, X_pred[i].shape[0]), dataset)[0]
            print(dataset.shape, X_pred[i].reshape(1, X_pred[i].shape[0]).shape)
            # Calculate the label of the datapoint
            mean_label = 1/self.k * labelset[distance.argsort()[:self.k]].sum()
            # Addiere 0.01 auf mean_label, damit bei 0.5 aufgerundet wird
            self.label_predict[i] = np.ceil(mean_label)

        return np.array(self.label_predict)

if __name__ == "__main__":
    knn = KNN(10)
    X, y = make_blobs(n_samples=10000, n_features=3, centers=2, random_state=20,
                      cluster_std=8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    knn.fit(X_train, y_train)
    label = knn.predict(X_test)
    print(len(X_test), X_test.shape)

    plt.scatter(X_test[:, 0], X_test[:, 1], c= label)
    plt.show()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    plt.show()
