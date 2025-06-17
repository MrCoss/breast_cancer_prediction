# breast_cancer_prediction.py

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Step 1: Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target
print("Feature names:", data.feature_names)
print("Target names:", data.target_names)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train Models
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

log_reg_model = LogisticRegression(max_iter=10000)
log_reg_model.fit(X_train, y_train)

# Step 5: Evaluate Models
print("\nSVM Accuracy:", accuracy_score(y_test, svm_model.predict(X_test)))
print("SVM Report:\n", classification_report(y_test, svm_model.predict(X_test)))

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_model.predict(X_test)))
print("Logistic Regression Report:\n", classification_report(y_test, log_reg_model.predict(X_test)))

# Step 6: Save Models
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(log_reg_model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler too!

# Step 7: Load Models & Predict on a Sample
svm_loaded = joblib.load('svm_model.pkl')
log_loaded = joblib.load('logistic_model.pkl')
scaler_loaded = joblib.load('scaler.pkl')

sample = X_test[0].reshape(1, -1)
sample_scaled = scaler_loaded.transform(sample)

svm_pred = svm_loaded.predict(sample_scaled)[0]
log_pred = log_loaded.predict(sample_scaled)[0]

print("\nSVM Sample Prediction:", data.target_names[svm_pred])
print("Logistic Regression Sample Prediction:", data.target_names[log_pred])

# Step 8: Visualize Class Distribution
labels = ['Benign', 'Malignant']
sizes = [sum(y == 1), sum(y == 0)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Distribution of Classes')
plt.axis('equal')
plt.show()
