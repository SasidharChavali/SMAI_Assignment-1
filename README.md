# Part 1: K-Nearest Neighbors (KNN) Algorithm with Hyperparameter Tuning and Vectorization

## In this section, we conduct an analysis of a given dataset and perform the following steps: 
## Data Visualization: We start by performing data visualization to gain insights into the dataset's characteristics.
## Label Distribution Analysis: We calculate and visualize the distribution of each label within the dataset, providing a comprehensive view of label frequencies.
## K-Nearest Neighbors (KNN) Algorithm Implementation: We proceed to implement the K-Nearest Neighbors (KNN) algorithm using the training dataset. KNN is a supervised machine learning algorithm used for classification tasks.
## Hyperparameter Tuning: To enhance the algorithm's performance, we conduct hyperparameter tuning. This involves systematically adjusting parameters (e.g., the number of neighbors) and evaluating the model's performance using appropriate metrics.
## Optimization via Vectorization: To optimize the computational efficiency of the KNN algorithm, we employ a vectorization technique. This technique leverages multi-threading to expedite distance computations, resulting in improved execution speed.

# Part 2: Multi-Label Classification with Decision Tree Classifiers and Hyperparameter Tuning

## In this section, we address a dataset where each data point can be associated with multiple labels. The following steps are performed:
## Label Analysis: We commence by identifying all distinct labels present in the dataset and provide a visualization of their distribution, shedding light on label frequencies.
## Power Set Creation and Feature Array Formation: Utilizing the distinct labels obtained, we create a powerset of labels. Subsequently, we construct a feature array based on this powerset.
## One-Hot Encoding: To prepare the feature array for modeling, we apply one-hot encoding to represent label combinations as binary features.
## Decision Tree Classifier for Multi-Label Classification: We build a decision tree classifier using the one-hot encoded feature array. This classifier is designed to handle multi-label classification tasks.
## Hyperparameter Tuning: To optimize the decision tree classifier's performance, we engage in hyperparameter tuning. This entails systematically adjusting parameters (e.g., tree depth) and assessing the model's effectiveness using evaluation metrics such as F1-score, accuracy, precision, and recall.
## Performance Metric Reporting: Finally, we report the performance metrics, including F1-score, accuracy, precision, and recall, for both the single-label and multi-label classification scenarios, providing insights into the models' effectiveness in handling multi-label data.
