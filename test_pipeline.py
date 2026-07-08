"""
Quick terminal test for the full loan default prediction pipeline.
Run with: python test_pipeline.py
"""

import joblib
import pandas as pd
import numpy as np

# ---- Load everything the pipeline needs ----
scaler = joblib.load('models/scaler.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')
best_model = joblib.load('models/best_model.pkl')
shap_explainer = joblib.load('models/shap_explainer.pkl')
features_a = joblib.load('models/features_a.pkl')
features_b = joblib.load('models/features_b.pkl')

numeric_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
categorical_cols = ['EmploymentType', 'HasCoSigner', 'Education', 'LoanPurpose',
                     'HasMortgage', 'HasDependents', 'MaritalStatus']

VALID_CATEGORIES = {
    'EmploymentType': ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'],
    'HasCoSigner': ['Yes', 'No'],
    'Education': ["High School", "Bachelor's", "Master's", "PhD"],
    'LoanPurpose': ['Auto', 'Business', 'Education', 'Home', 'Other'],
    'HasMortgage': ['Yes', 'No'],
    'HasDependents': ['Yes', 'No'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
}


def normalize(text):
    return ''.join(ch for ch in text.lower().strip() if ch.isalnum())


def match_category(user_value, column_name):
    valid_options = VALID_CATEGORIES[column_name]
    normalized_user_value = normalize(user_value)

    for option in valid_options:
        if normalize(option) == normalized_user_value:
            return option

    print(f"  '{user_value}' not recognized. Valid options: {', '.join(valid_options)}")
    retry = input(f"  Please re-enter {column_name}: ")
    return match_category(retry, column_name)


def get_user_input():
    print("\n--- Enter Borrower Details ---")
    print("(Categorical answers are not case-sensitive and punctuation doesn't matter)\n")

    data = {
        'Age': int(input("Age: ")),
        'Income': float(input("Income: ")),
        'LoanAmount': float(input("Loan Amount: ")),
        'CreditScore': int(input("Credit Score: ")),
        'MonthsEmployed': int(input("Months Employed: ")),
        'NumCreditLines': int(input("Number of Credit Lines: ")),
        'InterestRate': float(input("Interest Rate (e.g. 12.5): ")),
        'LoanTerm': int(input("Loan Term (months, e.g. 36): ")),
        'DTIRatio': float(input("DTI Ratio (e.g. 0.35): ")),
        'EmploymentType': match_category(
            input(f"Employment Type {VALID_CATEGORIES['EmploymentType']}: "), 'EmploymentType'),
        'HasCoSigner': match_category(
            input(f"Has Co-Signer? {VALID_CATEGORIES['HasCoSigner']}: "), 'HasCoSigner'),
        'Education': match_category(
            input(f"Education {VALID_CATEGORIES['Education']}: "), 'Education'),
        'LoanPurpose': match_category(
            input(f"Loan Purpose {VALID_CATEGORIES['LoanPurpose']}: "), 'LoanPurpose'),
        'HasMortgage': match_category(
            input(f"Has Mortgage? {VALID_CATEGORIES['HasMortgage']}: "), 'HasMortgage'),
        'HasDependents': match_category(
            input(f"Has Dependents? {VALID_CATEGORIES['HasDependents']}: "), 'HasDependents'),
        'MaritalStatus': match_category(
            input(f"Marital Status {VALID_CATEGORIES['MaritalStatus']}: "), 'MaritalStatus'),
    }
    return pd.DataFrame([data])


def run_pipeline(new_input_raw):
    # One-hot encode to match training format
    new_encoded = pd.get_dummies(new_input_raw, columns=categorical_cols)

    # Add any missing dummy columns as 0 (categories not present in this single row)
    for col in features_b:
        if col not in new_encoded.columns and col != 'Cluster_ID':
            new_encoded[col] = 0

    # Scale numeric columns using the fitted scaler
    new_encoded[numeric_cols] = scaler.transform(new_encoded[numeric_cols])

    # Align to Model A's column order first
    new_encoded_A = new_encoded[features_a]

    # Assign cluster
    cluster_id = kmeans.predict(new_encoded_A)[0]

    # Build Model B row
    new_encoded_B = new_encoded_A.copy()
    new_encoded_B['Cluster_ID'] = cluster_id
    new_encoded_B = new_encoded_B[features_b]

    # Predict
    prediction = best_model.predict(new_encoded_B)[0]
    probability = best_model.predict_proba(new_encoded_B)[0][1]

    # SHAP explanation
    shap_vals = shap_explainer.shap_values(new_encoded_B, check_additivity=False)
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[0, :, 1]
    else:
        shap_vals = shap_vals[0]

    contribution = pd.DataFrame({
        'Feature': new_encoded_B.columns,
        'Value': new_encoded_B.values[0],
        'SHAP Contribution': shap_vals
    }).sort_values(by='SHAP Contribution', key=abs, ascending=False)

    return cluster_id, prediction, probability, contribution


if __name__ == "__main__":
    new_input_raw = get_user_input()
    cluster_id, prediction, probability, contribution = run_pipeline(new_input_raw)

    print("\n" + "=" * 50)
    print(f"Assigned Cluster: {cluster_id}")
    print(f"Prediction: {'DEFAULT' if prediction == 1 else 'NO DEFAULT'}")
    print(f"Probability of Default: {probability:.2%}")
    print("=" * 50)
    print("\nTop feature contributions (SHAP):")
    print(contribution.head(10).to_string(index=False))
