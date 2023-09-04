import sys
import numpy as np
from knnoptimized import createKNNOptimized
from sklearn.model_selection import train_test_split

if len(sys.argv) != 2:
    print("Error: Input format is python eval.py <path to npy file>")
    sys.exit(1)

npy_file = sys.argv[1]

try:
    data = np.load('data.npy', allow_pickle=True)
    X = data[:, 1:3]
    y = data[:, 3]
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

    test_data = np.load(npy_file, allow_pickle=True)
    X_test = test_data[:, 1:3]
    y_test = test_data[:, 3]

    K = 13
    distance_metric = 'manhattan'
    encoder_type = 'VIT'

    kNN = createKNNOptimized(K, X_train, y_train, distance_metric, encoder_type)
    f1, accuracy, precision, recall = kNN.evaluate(X_test, y_test)

    print("Evaluation Scores:")
    print("Accuracy: ", accuracy)
    print("F1-Score: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)

except Exception as e:
    print("An error occurred:", e)
    sys.exit(1)
