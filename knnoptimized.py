import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class KNNClassifierOptimized:
    
    # k is the number of neighbors to consider
    # X_train is a list of tuples (resnet, vit)
    # y_train is a list of labels
    # distance_metric is the distance metric to use
    # encoder_type is the encoder type to use
    
    def __init__(self, k, X_train, y_train, distance_metric, encoder_type):
        self.k = k
        self.resnet_data = np.concatenate(X_train[:,0], axis=0)
        self.vit_data = np.concatenate(X_train[:,1], axis=0)
        self.y_train = y_train
        self.distance_metric = distance_metric
        self.encoder_type = encoder_type

    def distance(self, train_data, x):

        # Uses vectorization to calculate the distance between x and every element in train_data
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(train_data - x, axis=1)
        if self.distance_metric == 'manhattan':
            return np.linalg.norm(train_data - x, ord=1, axis=1)
        if self.distance_metric == 'cosine_distance':
            return 1 - np.dot(train_data, x)/(np.linalg.norm(train_data, axis=1)*np.linalg.norm(x))
    
    def predict_one(self, x, train_data):

        distances = self.distance(train_data, x)
        tuples = list(zip(distances, self.y_train))
        tuples.sort(key=lambda x: x[0])
        neighbors = tuples[:self.k]
        neighbor_labels = [label for (_, label) in neighbors]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        return most_common_label

    def predict(self, X):

        # Predicts the labels of X
        predictions = np.array([])
        resnet_val_data = np.concatenate(X[:, 0], axis=0)
        vit_val_data = np.concatenate(X[:, 1], axis=0)
        if(self.encoder_type == 'ResNet'):
            for x in resnet_val_data:
                predictions = np.append(predictions, self.predict_one(x, self.resnet_data))
        elif(self.encoder_type == 'VIT'):
            for x in vit_val_data:
                predictions = np.append(predictions, self.predict_one(x, self.vit_data))
        return predictions

    def evaluate(self, X_val, y_val):
        # Evaluates the model on X_val and y_val
        y_pred = self.predict(X_val)
        
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=True)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro', zero_division=True)
        recall = recall_score(y_val, y_pred, average='macro', zero_division=True)
        return f1, accuracy, precision, recall


def createKNNOptimized(k, X_train, y_train, distance_metric, encoder_type):
    kNN = KNNClassifierOptimized(k, X_train, y_train, distance_metric, encoder_type)
    return kNN