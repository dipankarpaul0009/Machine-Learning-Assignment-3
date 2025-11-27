# Machine-Learning-Assignment-3

Decision Tree Classification on the Iris Dataset

This project demonstrates how to build, train, visualize, and evaluate a Decision Tree Classifier using the classic Iris dataset.
The assignment includes data loading, preprocessing, model training, visualization using graphviz, and performance evaluation.

# Installation

Install all required libraries:

pip install scikit-learn pandas matplotlib graphviz


Note:
You must also install Graphviz on your system (not just the Python package).
Download: https://graphviz.org/download/

# Project Overview

This assignment covers:

Loading the Iris dataset

Splitting data into training & testing sets

Training a Decision Tree Classifier

Plotting the tree using Matplotlib

Exporting the tree visualization using Graphviz

Measuring accuracy and generating a classification report

# Code Explanation
1. Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import graphviz

#2. Loading the Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

#3. Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#4. Training the Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

#5. Making Predictions & Evaluating
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#6. Visualizing the Tree (Matplotlib)
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

#7. Exporting as Graphviz File
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")

#Results

Achieves high accuracy (typically 95%–100%)

Provides clear visualization of decision boundaries

Classification report includes:

Precision

Recall

F1-score

#Learning Outcomes

Understanding Decision Tree algorithms

Visualizing ML models using Graphviz

Evaluating classification models

Working with SciKit-Learn’s built-in datasets

Plotting and exporting decision trees

#File Structure
├── Decision-Tree.ipynb     # Jupyter notebook with full analysis
├── iris_decision_tree.pdf  # Exported decision tree (if generated)
├── README.md               # Project documentation

Submitted by,

Name :- Dipankar Paul
subject :- Machine Learning 
Assignment on – Decision Trees
Semester - 7th sem
branch- CSE
