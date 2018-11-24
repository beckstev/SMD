import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


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
        self.label_fit = []

        # Predicted label of the real dataset
        self.label_predict = []


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
        for i in range(X.shape[0]):
            self.label_prediction(X[i], X, y, self.label_fit)
        return self.label_fit


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
        for i in range(X_pred.shape[0]):
            self.label_prediction(X_pred[i], self.traning_dataset,
                                  np.array(self.label_fit), self.label_predict)

        return np.array(self.label_predict)

    def label_prediction(self, x_0, dataset, labelset, klist):
        # calculate the distance between every point and the test point
        distance = cdist(x_0.reshape(1, 2), dataset)[0]
        # Calculate the label of the datapoint
        mean_label = 1/self.k * labelset[distance.argsort()[-self.k:]].sum()

        # Addiere 0.01 auf mean_label, damit bei 0.5 aufgerundet wird
        klist.append(int(np.round(mean_label + 0.01)))


if __name__ == "__main__":
    knn = KNN(5)
    X, y = make_blobs(n_samples=500, n_features=2, centers=2, random_state=12,
                      cluster_std=1.4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    knn.fit(X_train, y_train)
    knn.predict(X_test)
    print(len(X_test), X_test.shape)
