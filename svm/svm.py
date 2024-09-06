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

    def evaluate(self, connected_components, labels):
        # Extract features
        X, _ = extract_features(connected_components)
        X_scaled = self.scaler.transform(X)

        # Evaluate the model
        y_pred = self.classifier.predict(X_scaled)
        print(classification_report(labels, y_pred))

    def classify(self, connected_component):
        # Extract features
        X, _ = extract_features([connected_component])
        X_scaled = self.scaler.transform(X)

        # Classify the connected component
        prediction = self.classifier.predict(X_scaled)[0]
        return prediction

# Usage example
if __name__ == "__main__":
    # Assume you have a list of connected components and their labels
    connected_components = [...]  # Your list of connected components
    labels = [...]  # Your list of labels (0 for normal text, 1 for math text)

    # Create and train the text classifier
    classifier = TextClassifier()
    classifier.train(connected_components, labels)

    # Evaluate the classifier
    classifier.evaluate(connected_components, labels)

    # Classify a new connected component
    new_cc = connected_components[0]
    prediction = classifier.classify(new_cc)
    print(f"Prediction: {'Math Text' if prediction == 1 else 'Normal Text'}")
