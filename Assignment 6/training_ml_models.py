import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)

param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("\nBest parameters for Random Forest (GridSearchCV):")
print(grid_rf.best_params_)
evaluate_model("Tuned Random Forest", grid_rf.best_estimator_, X_test, y_test)

param_dist_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

random_svm = RandomizedSearchCV(SVC(), param_distributions=param_dist_svm, n_iter=10, cv=5, scoring='f1', random_state=42, n_jobs=-1)
random_svm.fit(X_train, y_train)
print("\nBest parameters for SVM (RandomizedSearchCV):")
print(random_svm.best_params_)
evaluate_model("Tuned SVM", random_svm.best_estimator_, X_test, y_test)

model_scores = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    model_scores.append((name, f1))

model_scores.append(("Tuned Random Forest", f1_score(y_test, grid_rf.best_estimator_.predict(X_test))))
model_scores.append(("Tuned SVM", f1_score(y_test, random_svm.best_estimator_.predict(X_test))))

best_model = sorted(model_scores, key=lambda x: x[1], reverse=True)[0]
print(f"\nBest Model: {best_model[0]} with F1 Score: {best_model[1]:.4f}")
