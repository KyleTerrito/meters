from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score


from source.data_preprocess import DataPreprocessing



class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def ANN(self, X_train, X_test, y_train, y_test):
        #Create DT model
        MLP_classifier = MLPClassifier()

        #Train the model
        MLP_classifier.fit(X_train, y_train)

        #Test the model
        MLP_predicted = MLP_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(MLP_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, MLP_predicted)

        return MLP_predicted