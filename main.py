import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Function to evaluate a model
def evaluate_model(X_train, X_test, y_train, y_test, scaler, model, model_name):
    # Standardize the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()

    classes = np.unique(y_test)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return results

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    'Adaboost Classifier': AdaBoostClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'MLP Classifier': MLPClassifier(),
    'SVM': SVC(),
}

# Iterate through models
for model_name, model in models.items():
    print(f"\nEvaluating {model_name} on Balanced Dataset:")
    results_balanced = evaluate_model(X_train, X_test, y_train, y_test, StandardScaler(), model, model_name)
    print(f"{model_name} - {results_balanced}")

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the resampled dataset
    X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    print(f"\nEvaluating {model_name} on Unbalanced Dataset:")
    results_unbalanced = evaluate_model(X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled,
                                        StandardScaler(), model, model_name)
    print(f"{model_name} - {results_unbalanced}")
