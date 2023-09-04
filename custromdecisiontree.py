import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

class CustomDecisionTreeClassifier:
    def __init__(self, max_depth, max_features, criterion):
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, criterion=criterion)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        f1 = f1_score(y, y_pred, average='macro', zero_division=True)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro', zero_division=True)
        recall = recall_score(y, y_pred, average='macro', zero_division=True)
        cm = confusion_matrix(y.argmax(axis=1), y_pred.argmax(axis=1))
        return f1, accuracy, precision, recall, cm
    
    def get_params(self, deep=True):
        return {
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'criterion': self.criterion
        }