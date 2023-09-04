import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class KNNClassifierInitial:
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
    
    # Calculates the distance between x and every element in train_data
    def distance(self, train_data, x):
        distances = []
        if self.distance_metric == 'euclidean':
            for element in train_data:
                distances.append(np.linalg.norm(element - x))
        if self.distance_metric == 'manhattan':
            for element in train_data:
                distances.append(np.linalg.norm(element - x, ord=1))
        if self.distance_metric == 'cosine_distance':
            for element in train_data:
                distances.append(1 - np.dot(element, x)/(np.linalg.norm(element)*np.linalg.norm(x)))

        return distances
    
    # Predicts the label of x
    def predict_one(self, x, train_data):
        distances = self.distance(train_data, x)
        tuples = list(zip(distances, self.y_train))
        tuples.sort(key=lambda x: x[0])
        neighbors = tuples[:self.k]
        neighbor_labels = [label for (_, label) in neighbors]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        return most_common_label
    
    # Predicts the labels of X
    def predict(self, X):
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
    
    # Evaluates the model on X_val and y_val
    def evaluate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=True)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro', zero_division=True)
        recall = recall_score(y_val, y_pred, average='macro', zero_division=True)
        return f1, accuracy, precision, recall

def createKNNInitial(k, X_train, y_train, distance_metric, encoder_type):
    kNN = KNNClassifierInitial(k, X_train, y_train, distance_metric, encoder_type)
    return kNN