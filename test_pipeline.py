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


def get_float_input(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return float(value)
        except ValueError:
            print("  Please enter a valid number.")


def get_int_input(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return int(value)
        except ValueError:
            print("  Please enter a valid whole number.")


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
    print("\n--- Enter Applicant Details ---")

    age = get_int_input("Applicant's Age: ")
    income = get_float_input("Applicant's Monthly Income: ")
    loan_amount = get_float_input("Loan Amount requested: ")

    print("\nCredit Score reflects how reliably the applicant has repaid debts in the past.")
    print("Roughly: 300-579 = Poor, 580-669 = Fair, 670-739 = Good, 740-799 = Very Good, 800-850 = Excellent")
    credit_score = get_int_input("Applicant's Credit Score: ")

    months_employed = get_int_input("\nHow many months has the applicant been at their current job? ")

    print("\nCredit lines = the number of open credit accounts the applicant has (credit cards, personal loans,")
    print("auto loans, store cards, etc. all count as one line each).")
    num_credit_lines = get_int_input("Number of Credit Lines: ")

    
    interest_rate = get_float_input("Interest Rate for this loan offer (e.g. 12.5 for 12.5%): ")
    loan_term = get_int_input("Loan Term in months (e.g. 36): ")

    print("\nTo calculate the applicant's Debt-to-Income (DTI) Ratio, enter their existing monthly debt payments.")
    print("This includes any current loan payments, credit card minimums, etc. (not including this new loan).")
    monthly_debt = get_float_input("Applicant's total existing monthly debt payments: ")
    dti_ratio = monthly_debt / income if income > 0 else 0
    print(f"-> Calculated DTI Ratio: {dti_ratio:.2f} ({dti_ratio:.0%} of income goes to debt)")

    data = {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio,
        'EmploymentType': match_category(
            input(f"\nApplicant's Employment Type {VALID_CATEGORIES['EmploymentType']}: "), 'EmploymentType'),
        'HasCoSigner': match_category(
            input(f"Does the applicant have a Co-Signer? {VALID_CATEGORIES['HasCoSigner']}: "), 'HasCoSigner'),
        'Education': match_category(
            input(f"Applicant's Education {VALID_CATEGORIES['Education']}: "), 'Education'),
        'LoanPurpose': match_category(
            input(f"Loan Purpose (Auto = vehicle loan) {VALID_CATEGORIES['LoanPurpose']}: "), 'LoanPurpose'),
        'HasMortgage': match_category(
            input(f"Does the applicant have a Mortgage? {VALID_CATEGORIES['HasMortgage']}: "), 'HasMortgage'),
        'HasDependents': match_category(
            input(f"Does the applicant have Dependents? {VALID_CATEGORIES['HasDependents']}: "), 'HasDependents'),
        'MaritalStatus': match_category(
            input(f"Applicant's Marital Status {VALID_CATEGORIES['MaritalStatus']}: "), 'MaritalStatus'),
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
    proba_default = best_model.predict_proba(new_encoded_B)[0][1]

    # Show confidence in whichever label was actually predicted, not always P(Default)
    if prediction == 1:
        predicted_label = "DEFAULT"
        confidence = proba_default
    else:
        predicted_label = "NO DEFAULT"
        confidence = 1 - proba_default

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

    return cluster_id, predicted_label, confidence, contribution


def explain_in_plain_language(contribution_df, target_coverage=0.85):
    """
    Converts raw SHAP contributions into a human-readable summary.
    Includes as many top features as needed to cover roughly target_coverage
    (default 87%, within the 85-90% range) of the total influence, rather
    than a fixed number of features every time.
    """
    friendly_names = {
        'Age': 'Age',
        'Income': 'Income',
        'LoanAmount': 'Loan Amount',
        'CreditScore': 'Credit Score',
        'MonthsEmployed': 'Employment Duration',
        'NumCreditLines': 'Number of Credit Lines',
        'InterestRate': 'Interest Rate',
        'LoanTerm': 'Loan Term',
        'DTIRatio': 'Debt-to-Income Ratio',
        'Cluster_ID': 'Borrower Segment',
        'HasCoSigner_Yes': 'Having a Co-Signer',
        'HasMortgage_Yes': 'Having a Mortgage',
        'HasDependents_Yes': 'Having Dependents',
        'MaritalStatus_Married': 'Being Married',
        'MaritalStatus_Single': 'Being Single',
        'EmploymentType_Part-time': 'Part-time Employment',
        'EmploymentType_Self-employed': 'Self-employment',
        'EmploymentType_Unemployed': 'Unemployment',
        "Education_Master's": "Having a Master's Degree",
        'Education_PhD': 'Having a PhD',
        'Education_High School': 'High School Education',
        'LoanPurpose_Business': 'Business Loan Purpose',
        'LoanPurpose_Education': 'Education Loan Purpose',
        'LoanPurpose_Home': 'Home Loan Purpose',
        'LoanPurpose_Other': 'Other Loan Purpose',
    }

    total_influence = contribution_df['SHAP Contribution'].abs().sum()

    sorted_df = contribution_df.reindex(
        contribution_df['SHAP Contribution'].abs().sort_values(ascending=False).index
    ).copy()
    sorted_df['Share'] = (sorted_df['SHAP Contribution'].abs() / total_influence)
    sorted_df['CumulativeShare'] = sorted_df['Share'].cumsum()

    # Include features up to and including the first one that reaches target_coverage
    cutoff_idx = sorted_df[sorted_df['CumulativeShare'] >= target_coverage].index
    if len(cutoff_idx) > 0:
        first_cutoff = sorted_df.index.get_loc(cutoff_idx[0])
        top_features = sorted_df.iloc[:first_cutoff + 1]
    else:
        top_features = sorted_df  # fallback: show all if coverage never reached

    print(f"\nThe prediction was mainly shaped by the following {len(top_features)} factor(s), "
          f"covering about {top_features['Share'].sum():.0%} of the total influence:\n")

    for _, row in top_features.iterrows():
        name = friendly_names.get(row['Feature'], row['Feature'])
        direction = "increased" if row['SHAP Contribution'] > 0 else "decreased"
        print(f"  - {name}: {direction} default risk (about {row['Share']*100:.0f}% of the total influence)")

    remaining_share = 1 - top_features['Share'].sum()
    if remaining_share > 0.005:
        print(f"\n  All other factors combined accounted for the remaining {remaining_share:.0%} of influence.")


if __name__ == "__main__":
    new_input_raw = get_user_input()
    cluster_id, predicted_label, confidence, contribution = run_pipeline(new_input_raw)

    print("\n" + "=" * 50)
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print("=" * 50)

    explain_in_plain_language(contribution, target_coverage=0.87)