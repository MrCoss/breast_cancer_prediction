# Breast Cancer Prediction using Classic ML

[](https://shields.io/)
[](https://www.python.org/)
[](https://scikit-learn.org/)
[](https://opensource.org/licenses/MIT)

A foundational machine learning project that classifies breast cancer tumors as **benign** or **malignant**. This project implements and compares two robust classification algorithms, **Support Vector Machine (SVM)** and **Logistic Regression**, on the Breast Cancer Wisconsin (Diagnostic) dataset.

-----

## Table of Contents

  - [1. Project Context & Objective](https://www.google.com/search?q=%231-project-context--objective)
  - [2. The Machine Learning Workflow](https://www.google.com/search?q=%232-the-machine-learning-workflow)
  - [3. Machine Learning Models](https://www.google.com/search?q=%233-machine-learning-models)
  - [4. Project Structure Explained](https://www.google.com/search?q=%234-project-structure-explained)
  - [5. Technical Stack](https://www.google.com/search?q=%235-technical-stack)
  - [6. Local Setup & Execution Guide](https://www.google.com/search?q=%236-local-setup--execution-guide)
  - [7. Sample Output](https://www.google.com/search?q=%237-sample-output)
  - [8. Author & Contact](https://www.google.com/search?q=%238-author--contact)

-----

## 1\. Project Context & Objective

Breast cancer is one of the most common cancers worldwide, and early and accurate diagnosis is crucial for improving patient outcomes. Diagnostic procedures often involve analyzing the characteristics of a tumor. Machine learning offers a powerful tool to automate this analysis and provide a reliable, data-driven classification.

The objective of this project is to build and evaluate a binary classification system that can distinguish between benign (non-cancerous) and malignant (cancerous) tumors based on their measured features. This serves as a practical demonstration of a complete ML pipeline, from data preprocessing to model training, evaluation, and persistence.

-----

## 2\. The Machine Learning Workflow

The project follows a standard, structured workflow to ensure reproducibility and reliable results.

### Step 1: Data Loading & Exploration

  - The **Breast Cancer Wisconsin (Diagnostic) Dataset** is loaded directly from `sklearn.datasets`.
  - An initial analysis is performed to understand the feature set (e.g., radius, texture, smoothness) and to visualize the class distribution between benign and malignant tumors.

### Step 2: Data Preprocessing

  - **Train-Test Split:** The dataset is divided into a training set (80%) and a testing set (20%) using `train_test_split` to ensure the model is evaluated on unseen data.
  - **Feature Scaling:** A `StandardScaler` is fitted on the training data to normalize the features. Scaling is essential for distance-based algorithms like SVM to perform optimally. The same scaler is used to transform the test data to prevent data leakage.

### Step 3: Model Training

  - Two distinct classification models are trained on the preprocessed training data:
    1.  **Support Vector Machine (SVM):** A powerful model that finds the optimal hyperplane to separate the two classes.
    2.  **Logistic Regression:** A robust and interpretable linear model that calculates the probability of a tumor being malignant.

### Step 4: Model Evaluation

  - The performance of each trained model is assessed on the unseen test set.
  - The primary metric used is **accuracy**, which measures the percentage of correct predictions.
  - Further evaluation can be done using a **classification report** (to see precision, recall, and F1-score) and a **confusion matrix**.

### Step 5: Model Persistence

  - The trained SVM model, Logistic Regression model, and the `StandardScaler` object are saved to disk as `.pkl` files using `joblib`.
  - This allows the models to be reloaded and used for future predictions without needing to retrain them, a critical step for deployment.

-----

## 3\. Machine Learning Models

  - **Support Vector Machine (SVM):** Chosen for its effectiveness in high-dimensional spaces and its ability to model non-linear decision boundaries using different kernels (though a linear kernel is used here). It is a powerful "black box" model for classification.
  - **Logistic Regression:** Selected for its simplicity, speed, and high interpretability. It serves as an excellent baseline model and performs very well on linearly separable data.

-----

## 4\. Project Structure Explained

The repository is organized with a focus on simplicity and functionality.

```
.
├── bcp.py                # The main Python script containing the entire ML workflow.
├── svm_model.pkl         # The serialized, trained Support Vector Machine model.
├── logistic_model.pkl    # The serialized, trained Logistic Regression model.
├── scaler.pkl            # The saved StandardScaler object used for data normalization.
├── .gitignore            # Specifies files for Git to ignore (e.g., __pycache__).
└── README.md             # This detailed project documentation.
```

-----

## 5\. Technical Stack

  - **Core Language:** Python
  - **Machine Learning:** Scikit-learn
  - **Data Visualization:** Matplotlib
  - **Model Persistence:** Joblib
  - **Data Handling:** NumPy

-----

## 6\. Local Setup & Execution Guide

To run this project on your local machine, follow these steps.

1.  **Prerequisites:** Ensure you have Python 3.7+ installed.

2.  **Clone the Repository (Optional):** If you have the `bcp.py` script, you can skip this.

    ```bash
    git clone https://github.com/MrCoss/Breast-Cancer-Prediction.git
    cd Breast-Cancer-Prediction
    ```

3.  **Set Up a Virtual Environment** (Recommended):

    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**

    ```bash
    pip install scikit-learn matplotlib joblib
    ```

5.  **Run the Script:**
    Executing the script will perform all steps: data loading, preprocessing, training, evaluation, and saving the models.

    ```bash
    python bcp.py
    ```

-----

## 7\. Sample Output

After running the script, you can expect to see the following output in your terminal, indicating the performance of each model on the test data.

  - **SVM Accuracy:** Approximately 95.6%
  - **Logistic Regression Accuracy:** Approximately 97.4%
  - **Generated Files:** `svm_model.pkl`, `logistic_model.pkl`, and `scaler.pkl` will be created in your directory.

-----

## 8\. Author & Contact

This project was built by **Costas Pinto** for learning and deployment practice.

  - **GitHub:** [MrCoss](https://github.com/MrCoss)
