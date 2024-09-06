import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from preprocessing.segmentation.connected_component_new import extract_features

class TextClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = None

        self.feature_names = []


    ''''
    There is a lack of labeled data, the plan is to manually
    label some data to train a SVM model. The hope is that the 
    amount of data needed to train the model is less than 10^3.
    '''

    def train(self, connected_components, labels):
        # Extract features
        x, self.feature_names = extract_features(connected_components)

        # Scaled features
        x_scaled = self.scaler.fit_transform(x)

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }

        # Create and train the SVM classifier
        self.classifier = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1)
        self.classifier.fit(x_scaled, labels)

        print("Best parameters:", self.classifier.best_params_)
