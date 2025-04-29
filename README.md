# 🕵️‍♀️ Fraud Detection Model

## 📌 Overview

This project implements a machine learning model to detect fraudulent financial transactions. The model is trained on a dataset containing transaction details and uses ensemble methods — **Random Forest**, **XGBoost**, and **LightGBM** — to classify transactions as fraudulent or legitimate. The final tuned **XGBoost** model achieves high performance with an **ROC-AUC score of 0.99921**.

---

## ✨ Key Features

- **Data Preprocessing**: Handles missing values, filters high-risk transaction types (`TRANSFER`, `CASH_OUT`), and encodes categorical variables.
- **Class Imbalance Handling**: Uses **SMOTE** to balance the dataset.
- **Model Training**: Evaluates **Random Forest**, **XGBoost**, and **LightGBM**, with **XGBoost** selected as the best performer.
- **Hyperparameter Tuning**: Uses **GridSearchCV** to optimize XGBoost parameters.
- **Threshold Tuning**: Adjusts the decision threshold to maximize recall for fraud detection.
- **Feature Importance Analysis**: Identifies key predictors of fraudulent transactions.
- **Deployment-Ready**: Includes scripts to save, load, and use the trained model.

---

## 🛠️ Installation

Clone the repository:
```
git clone <repository-url>
cd Fraud-detection
Install the required dependencies:

pip install -r requirements.txt
``` 

## 🚀 Usage
**🔹 Data Preparation**
Place your transaction data in a CSV file named Fraud.csv in the project directory.

Run the Jupyter notebook or Python script to preprocess the data and train the model.

**🔹 Training the Model**
Execute the provided notebook or script to:

Load and preprocess the data.

Train and evaluate the models.

Save the best model as fraud_detection_model.pkl.

**🔹 Making Predictions**
Use the saved model to predict fraud on new data

## 📈 Model Performance
ROC-AUC Score: 0.99921

**🔸 Confusion Matrix (Adjusted Threshold):**

- **Predicted Negative	Predicted Positive
- **Actual Negative	64,511	568
- **Actual Positive	3	191
- **Recall (Fraud Class): 98.45%

- **Precision (Fraud Class): 25.16%

## 🔍 Key Findings
**🔑 Top Fraud Predictors:**
newbalanceOrig: Remaining balance in the originator's account.

oldbalanceOrg: Originator's balance before the transaction.

newbalanceDest: Destination account's balance after the transaction.

amount: Transaction amount.

type: Transaction type (e.g., TRANSFER or CASH_OUT).

## 🔒 Prevention Recommendations:
Monitor high-risk transactions (TRANSFER, CASH_OUT) in real-time.

Flag transactions with unusual balance patterns or large amounts.

Implement multi-factor authentication for large transactions.

Regularly update the model with new data to adapt to evolving fraud patterns.

## 📁 Files
Fraud_Detection_Model.ipynb: Jupyter notebook containing the full workflow.

fraud_detection_model.pkl: Saved trained model.

requirements.txt: List of required Python packages.

## 🧪 Dependencies
- **Python 3.7+

- **Libraries:

- **pandas

- **numpy

- **scikit-learn

- **xgboost

- **lightgbm

- **imbalanced-learn

- **matplotlib

- **seaborn

## 👤 Author
Shane (Replace with your full name and contact info)

