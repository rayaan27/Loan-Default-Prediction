"""
Loan Default Prediction - Streamlit App
Run with: streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Loan Default Risk Predictor", layout="wide")

# ---------------- Load everything once ----------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('models/scaler.pkl')
    kmeans = joblib.load('models/kmeans_model.pkl')
    best_model = joblib.load('models/best_model.pkl')
    features_a = joblib.load('models/features_a.pkl')
    features_b = joblib.load('models/features_b.pkl')

    # FIX: best_model is a CalibratedClassifierCV wrapper around XGBoost.
    # shap.TreeExplainer cannot read the wrapper directly (InvalidModelError),
    # so we pull out the underlying tree model and explain that instead.
    # Calibration is a monotonic post-processing step, so relative feature
    # importance from the base model remains meaningful.
    try:
        base_model = best_model.calibrated_classifiers_[0].estimator
    except AttributeError:
        # older scikit-learn versions name this attribute differently
        base_model = best_model.calibrated_classifiers_[0].base_estimator

    shap_explainer = shap.TreeExplainer(base_model, feature_perturbation='tree_path_dependent')

    return scaler, kmeans, best_model, shap_explainer, features_a, features_b


@st.cache_resource
def load_decision_thresholds():
    # FIX: three-tier Approve/Review/Deny cutoffs, produced by threshold tuning.
    # Falls back to sane defaults if the file is missing so the app doesn't crash.
    try:
        with open('models/decision_threshold.json', 'r') as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        thresholds = {"review_threshold": 0.10, "deny_threshold": 0.18}
    return thresholds


@st.cache_data
def load_data():
    df_clusters = pd.read_csv('data/preprocessed_data_with_clusters.csv')
    comparison_df = pd.read_csv('data/model_comparison.csv')
    return df_clusters, comparison_df


@st.cache_resource
def load_global_shap():
    shap_vals_global = np.load('models/shap_values_summary.npy')
    shap_sample = pd.read_csv('data/shap_summary_sample.csv')
    shap_vals_global = shap_vals_global[:len(shap_sample)]
    return shap_vals_global, shap_sample


scaler, kmeans, best_model, shap_explainer, features_a, features_b = load_artifacts()
decision_thresholds = load_decision_thresholds()
df_clusters, comparison_df = load_data()

REVIEW_THRESHOLD = decision_thresholds.get("review_threshold", 0.3)
DENY_THRESHOLD = decision_thresholds.get("deny_threshold", 0.6)

numeric_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
categorical_cols = ['EmploymentType', 'HasCoSigner', 'Education', 'LoanPurpose',
                     'HasMortgage', 'HasDependents', 'MaritalStatus']

FRIENDLY_NAMES = {
    'Age': 'Age', 'Income': 'Income', 'LoanAmount': 'Loan Amount',
    'CreditScore': 'Credit Score', 'MonthsEmployed': 'Employment Duration',
    'NumCreditLines': 'Number of Credit Lines', 'InterestRate': 'Interest Rate',
    'LoanTerm': 'Loan Term', 'DTIRatio': 'Debt-to-Income Ratio',
    'Cluster_ID': 'Applicant Group', 'HasCoSigner_Yes': 'Having a Co-Signer',
    'HasMortgage_Yes': 'Having a Mortgage', 'HasDependents_Yes': 'Having Dependents',
    'MaritalStatus_Married': 'Being Married', 'MaritalStatus_Single': 'Being Single',
    'EmploymentType_Part-time': 'Part-time Employment',
    'EmploymentType_Self-employed': 'Self-employment',
    'EmploymentType_Unemployed': 'Unemployment',
    "Education_Master's": "Having a Master's Degree", 'Education_PhD': 'Having a PhD',
    'Education_High School': 'High School Education',
    'LoanPurpose_Business': 'Business Loan Purpose', 'LoanPurpose_Education': 'Education Loan Purpose',
    'LoanPurpose_Home': 'Home Loan Purpose', 'LoanPurpose_Other': 'Other Loan Purpose',
}

# Binary/categorical one-hot columns need value-aware labels: (label if value==1, label if value==0)
BINARY_FEATURE_LABELS = {
    'HasCoSigner_Yes': ("Having a Co-Signer", "Not having a Co-Signer"),
    'HasMortgage_Yes': ("Having a Mortgage", "Not having a Mortgage"),
    'HasDependents_Yes': ("Having Dependents", "Not having Dependents"),
    'MaritalStatus_Married': ("Being Married", "Not being Married"),
    'MaritalStatus_Single': ("Being Single", "Not being Single"),
    "Education_Master's": ("Having a Master's Degree", "Not having a Master's Degree"),
    'Education_PhD': ("Having a PhD", "Not having a PhD"),
    'Education_High School': ("Having only a High School Education", "Having education beyond High School"),
    'LoanPurpose_Business': ("A Business Loan Purpose", "Not a Business Loan Purpose"),
    'LoanPurpose_Education': ("An Education Loan Purpose", "Not an Education Loan Purpose"),
    'LoanPurpose_Home': ("A Home Loan Purpose", "Not a Home Loan Purpose"),
    'LoanPurpose_Other': ("An 'Other' Loan Purpose", "Not an 'Other' Loan Purpose"),
    'EmploymentType_Part-time': ("Part-time Employment", "Not Part-time Employment"),
    'EmploymentType_Self-employed': ("Self-employment", "Not Self-employed"),
    'EmploymentType_Unemployed': ("Unemployment", "Not Unemployed"),
}


def get_feature_label(feature, value):
    """Returns a value-aware label for one-hot features, or the plain friendly name otherwise."""
    if feature in BINARY_FEATURE_LABELS:
        label_if_1, label_if_0 = BINARY_FEATURE_LABELS[feature]
        return label_if_1 if value == 1 else label_if_0
    return FRIENDLY_NAMES.get(feature, feature)


def get_decision(proba_default):
    """FIX: three-tier Approve/Review/Deny decision based on tuned thresholds."""
    if proba_default >= DENY_THRESHOLD:
        return "DENY"
    elif proba_default >= REVIEW_THRESHOLD:
        return "REVIEW"
    else:
        return "APPROVE"


# Clearer segment naming + a plain description of what it means for the applicant
CLUSTER_INFO = {
    0: ("Established, Short-Term Borrower",
        "Applicants like this tend to be older and prefer shorter repayment periods. "
        "This group has historically shown the lowest default rate."),
    1: ("Younger, Multi-Credit Borrower",
        "Applicants like this tend to be younger and manage several open credit accounts at once. "
        "This group has historically shown the highest default rate."),
    2: ("Established, Long-Term Borrower",
        "Applicants like this tend to be older and choose longer repayment periods. "
        "This group has historically shown a low default rate."),
    3: ("Younger, Fewer-Credit Borrower",
        "Applicants like this tend to be younger with fewer open credit accounts. "
        "This group has historically shown an elevated default rate."),
}

INPUT_DEFAULTS = {
    "applicant_name": "", "age": 0, "income": 0.0, "loan_amount": 0.0, "credit_score": 0,
    "months_employed": 0, "num_credit_lines": 0, "interest_rate": 0.0, "loan_term": 0,
    "monthly_debt": 0.0, "employment_type": "Full-time", "education": "High School",
    "loan_purpose": "Auto", "has_cosigner": "No", "has_mortgage": "No",
    "has_dependents": "No", "marital_status": "Single",
}


# ==================== PAGE 1: PREDICT ====================
def predict_page():
    # Streamlit deletes a widget's session_state entry whenever that widget doesn't
    # render during a script run — which happens every time we're on a different page
    # (st.navigation only runs the active page's function). So the widget keys
    # themselves can't be trusted to survive a round trip to Model Insights and back.
    # 'input_shadow' is a plain dict (not a widget key), so it's untouched by that
    # cleanup — we seed the widgets from it, and refresh it after each render below.
    if 'input_shadow' not in st.session_state:
        st.session_state['input_shadow'] = dict(INPUT_DEFAULTS)
    for k in INPUT_DEFAULTS:
        if k not in st.session_state:
            st.session_state[k] = st.session_state['input_shadow'][k]

    st.title("Loan Default Risk Predictor")
    st.caption("Enter applicant details below to predict default risk.")

    applicant_name = st.text_input("Applicant Name (optional, used only for saving this record)",
                                     key="applicant_name")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Applicant's Age", min_value=0, max_value=100, key="age")
        income = st.number_input("Applicant's Monthly Income", min_value=0.0, step=100.0, key="income")
        loan_amount = st.number_input("Loan Amount Requested", min_value=0.0, step=500.0, key="loan_amount")
        credit_score = st.number_input(
            "Applicant's Credit Score", min_value=0, max_value=850, key="credit_score",
            help="300-579 Poor | 580-669 Fair | 670-739 Good | 740-799 Very Good | 800-850 Excellent. "
                 "Leave at 0 if unknown.")

    with col2:
        months_employed = st.number_input(
            "Months at Current Job", min_value=0, key="months_employed",
            help="How long the applicant has held their CURRENT job specifically, in months — "
                 "not their total career length across all past jobs.")
        num_credit_lines = st.number_input(
            "Number of Credit Lines", min_value=0, key="num_credit_lines",
            help="The number of open credit accounts the applicant has — credit cards, personal loans, "
                 "auto loans, store cards, etc. all count as one line each.")
        interest_rate = st.number_input(
            "Interest Rate Offered (%)", min_value=0.0, max_value=50.0, step=0.1, key="interest_rate",
            help="The interest rate being offered on this loan. Set by the lender, not the applicant.")
        loan_term = st.number_input(
            "Loan Term (months)", min_value=0, max_value=360, step=1, key="loan_term",
            help="How many months the applicant will take to repay the loan. Enter any value, e.g. 18, 36, 60.")

    with col3:
        monthly_debt = st.number_input(
            "Applicant's Existing Monthly Debt Payments", min_value=0.0, step=50.0, key="monthly_debt",
            help="Total of the applicant's current monthly debt obligations (other loans, credit card "
                 "minimums, etc.), not including this new loan. Used to calculate DTI Ratio below.")
        dti_ratio = monthly_debt / income if income > 0 else 0
        st.metric(
            "Calculated DTI Ratio",
            f"{dti_ratio:.2f}",
            f"{dti_ratio:.0%} of income",
            help="Debt-to-Income Ratio: the share of monthly income already going toward debt payments. "
                 "Calculated automatically as Existing Monthly Debt ÷ Monthly Income. Lower is generally safer.")

        employment_type = st.selectbox("Employment Type", ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'],
                                        key="employment_type")
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"], key="education")
        loan_purpose = st.selectbox("Loan Purpose", ['Auto', 'Business', 'Education', 'Home', 'Other'],
                                     key="loan_purpose", help="What the loan will be used for. Auto = vehicle loan.")

    col4, col5, col6 = st.columns(3)
    with col4:
        has_cosigner = st.radio("Has Co-Signer?", ['No', 'Yes'], horizontal=True, key="has_cosigner",
                                 help="A co-signer is a second person who agrees to repay the loan if the "
                                      "applicant cannot, reducing risk to the lender.")
    with col5:
        has_mortgage = st.radio("Has Mortgage?", ['No', 'Yes'], horizontal=True, key="has_mortgage")
    with col6:
        has_dependents = st.radio("Has Dependents?", ['No', 'Yes'], horizontal=True, key="has_dependents")

    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'], key="marital_status")

    # Refresh the shadow copy with this run's current values, so they're preserved
    # even if the user navigates away before Streamlit clears the widget keys.
    st.session_state['input_shadow'] = {
        'applicant_name': applicant_name, 'age': age, 'income': income, 'loan_amount': loan_amount,
        'credit_score': credit_score, 'months_employed': months_employed,
        'num_credit_lines': num_credit_lines, 'interest_rate': interest_rate, 'loan_term': loan_term,
        'monthly_debt': monthly_debt, 'employment_type': employment_type, 'education': education,
        'loan_purpose': loan_purpose, 'has_cosigner': has_cosigner, 'has_mortgage': has_mortgage,
        'has_dependents': has_dependents, 'marital_status': marital_status,
    }

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
    predict_clicked = btn_col1.button("Predict Default Risk", type="primary")
    reset_clicked = btn_col2.button("Reset")

    if reset_clicked:
        # Cannot assign directly to st.session_state[k] here, because the widgets with
        # these keys have already been instantiated earlier in this run. Popping the key
        # removes it from session_state; the seeding loop at the top of this function
        # will then re-create it with its default value on the next run (st.rerun()).
        for k in INPUT_DEFAULTS:
            st.session_state.pop(k, None)
        st.session_state['input_shadow'] = dict(INPUT_DEFAULTS)
        st.session_state.pop('prediction_result', None)
        st.rerun()

    if predict_clicked:
        new_input_raw = pd.DataFrame([{
            'Age': age, 'Income': income, 'LoanAmount': loan_amount, 'CreditScore': credit_score,
            'MonthsEmployed': months_employed, 'NumCreditLines': num_credit_lines,
            'InterestRate': interest_rate, 'LoanTerm': loan_term, 'DTIRatio': dti_ratio,
            'EmploymentType': employment_type, 'HasCoSigner': has_cosigner, 'Education': education,
            'LoanPurpose': loan_purpose, 'HasMortgage': has_mortgage,
            'HasDependents': has_dependents, 'MaritalStatus': marital_status,
        }])

        new_encoded = pd.get_dummies(new_input_raw, columns=categorical_cols)
        for col in features_b:
            if col not in new_encoded.columns and col != 'Cluster_ID':
                new_encoded[col] = 0
        new_encoded[numeric_cols] = scaler.transform(new_encoded[numeric_cols])
        new_encoded_A = new_encoded[features_a]

        cluster_id = kmeans.predict(new_encoded_A)[0]
        new_encoded_B = new_encoded_A.copy()
        new_encoded_B['Cluster_ID'] = cluster_id
        new_encoded_B = new_encoded_B[features_b]

        proba_default = best_model.predict_proba(new_encoded_B)[0][1]

        # FIX: three-tier Approve/Review/Deny decision instead of binary DEFAULT/NO DEFAULT
        decision = get_decision(proba_default)
        confidence = proba_default if decision == "DENY" else 1 - proba_default

        shap_vals = shap_explainer.shap_values(new_encoded_B, check_additivity=False)
        shap_vals = np.array(shap_vals)
        shap_vals = shap_vals[0, :, 1] if shap_vals.ndim == 3 else shap_vals[0]

        contribution_df = pd.DataFrame({
            'Feature': new_encoded_B.columns,
            'Value': new_encoded_B.values[0],
            'SHAP Contribution': shap_vals
        })
        contribution_df = contribution_df.reindex(
            contribution_df['SHAP Contribution'].abs().sort_values(ascending=False).index)

        total_influence = contribution_df['SHAP Contribution'].abs().sum()
        contribution_df['Share'] = contribution_df['SHAP Contribution'].abs() / total_influence
        contribution_df['CumulativeShare'] = contribution_df['Share'].cumsum()

        cutoff_idx = contribution_df[contribution_df['CumulativeShare'] >= 0.87].index
        if len(cutoff_idx) > 0:
            first_cutoff = contribution_df.index.get_loc(cutoff_idx[0])
            top_features = contribution_df.iloc[:first_cutoff + 1]
        else:
            top_features = contribution_df

        # Store everything needed to redraw the result, so it survives reruns on this page
        st.session_state['prediction_result'] = {
            'applicant_name': applicant_name,
            'decision': decision,
            'proba_default': proba_default,
            'confidence': confidence,
            'cluster_id': int(cluster_id),
            'top_features': top_features.to_dict('records'),
        }

    # ---- Render result from session_state ----
    if 'prediction_result' in st.session_state:
        result = st.session_state['prediction_result']

        st.divider()
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            # FIX: three-tier decision display
            if result['decision'] == "DENY":
                st.error(f"### Decision: {result['decision']}")
            elif result['decision'] == "REVIEW":
                st.warning(f"### Decision: {result['decision']}")
            else:
                st.success(f"### Decision: {result['decision']}")
            st.metric("Probability of Default", f"{result['proba_default']:.1%}")
        
        

        st.subheader("Why this decision?")
        total_share = sum(f['Share'] for f in result['top_features'])
        st.caption(f"The following {len(result['top_features'])} factor(s) account for about "
                   f"{total_share:.0%} of the model's decision.")

        for row in result['top_features']:
            name = get_feature_label(row['Feature'], row['Value'])
            direction = "increased" if row['SHAP Contribution'] > 0 else "decreased"
            icon = "🔺" if direction == "increased" else "🔻"
            st.write(f"{icon} **{name}** — {direction} default risk (~{row['Share']*100:.0f}% of influence)")

        st.divider()
        if st.button("Save Record"):
            record = {
                'Applicant Name': result['applicant_name'] if result['applicant_name'] else "Unnamed",
                'Decision': result['decision'],
                'Confidence': f"{result['confidence']:.1%}",
                'Applicant Group': CLUSTER_INFO.get(result['cluster_id'], (f"Group {result['cluster_id']}",))[0],
                'Top Factors': "; ".join(
                    f"{get_feature_label(f['Feature'], f['Value'])} "
                    f"({'increased' if f['SHAP Contribution'] > 0 else 'decreased'} risk, "
                    f"{f['Share']*100:.0f}%)"
                    for f in result['top_features']
                ),
            }
            log_path = 'data/applicant_records.csv'
            try:
                existing = pd.read_csv(log_path)
                updated = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
            except FileNotFoundError:
                updated = pd.DataFrame([record])
            updated.to_csv(log_path, index=False)
            st.success(f"Saved record for {record['Applicant Name']}.")

    st.divider()
    with st.expander("View Saved Applicant Records"):
        try:
            saved_records = pd.read_csv('data/applicant_records.csv')
        except FileNotFoundError:
            saved_records = pd.DataFrame()

        if saved_records.empty:
            st.caption("No records saved yet.")
        else:
            for idx in reversed(saved_records.index):
                row = saved_records.loc[idx]
                # FIX: old rows saved before this change used a 'Prediction' column
                # instead of 'Decision'. Fall back gracefully instead of KeyError-ing.
                decision_label = row['Decision'] if 'Decision' in saved_records.columns and pd.notna(row.get('Decision')) \
                    else row.get('Prediction', 'N/A')
                card_col1, card_col2 = st.columns([5, 1])
                with card_col1:
                    st.markdown(f"**{row['Applicant Name']}** — {decision_label} ({row['Confidence']})")
                    st.caption(f"Applicant Group: {row['Applicant Group']}")
                    for factor in str(row['Top Factors']).split("; "):
                        st.markdown(f"- {factor}")
                with card_col2:
                    if st.button("Delete", key=f"delete_{idx}"):
                        saved_records = saved_records.drop(index=idx)
                        saved_records.to_csv('data/applicant_records.csv', index=False)
                        st.rerun()
                st.divider()


# ==================== PAGE 2: MODEL INSIGHTS ====================
def insights_page():
    st.title("Behind the Model")
    st.caption("How this tool works: the pipeline, its performance, and what drives its decisions.")

    best_row = comparison_df.loc[comparison_df['F1'].idxmax()]

    st.subheader("Deployed Model Performance")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Accuracy", f"{best_row['Accuracy']:.1%}",
                help="Percentage of all predictions (default and no-default) that were correct.")
    kpi2.metric("Precision", f"{best_row['Precision']:.1%}",
                help="Of everyone flagged as likely to default, the percentage who actually did.")
    kpi3.metric("Recall", f"{best_row['Recall']:.1%}",
                help="Of everyone who actually defaulted, the percentage the model correctly caught.")
    kpi4.metric("F1-Score", f"{best_row['F1']:.1%}",
                help="A balanced score combining Precision and Recall — used to select the best model "
                     "given the imbalance between defaulters and non-defaulters in this dataset.")
    st.caption(f"Deployed model: **{best_row['Model']} ({best_row['Variant']})** — "
               f"selected for highest F1-score across all classifiers tested.")

    st.divider()

    # Headers for both columns first, so both charts below start at the same height
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("With vs Without Clustering")
        st.caption("Does adding the applicant-group feature improve each classifier?")
    with col2:
        st.subheader("Default Rate by Applicant Group")
        st.caption("How does the historical default rate compare across applicant segments?")

    # Shared control row (applies to the left chart) — kept outside the columns so it
    # doesn't push the left chart down relative to the right one
    available_models = comparison_df['Model'].unique().tolist()
    selected_model = st.radio("Select a model to compare:", available_models, horizontal=True,
                               key="insights_model_select")

    col1, col2 = st.columns([1, 1])

    with col1:
        model_subset = comparison_df[comparison_df['Model'] == selected_model].copy()
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        model_subset.set_index('Variant')[['Accuracy', 'Precision', 'Recall', 'F1']].plot(kind='bar', ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, ha='center')
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.0)
        ax2.set_title(f"{selected_model}: With vs Without Cluster_ID")
        ax2.legend(loc='upper left', fontsize=8, ncol=2, frameon=False)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col2:
        default_rate_by_cluster = df_clusters.groupby('Cluster_ID')['Default'].mean().sort_values(ascending=False)
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        labels = [CLUSTER_INFO.get(i, (f"Group {i}",))[0] for i in default_rate_by_cluster.index]
        ax1.barh(labels, default_rate_by_cluster.values * 100, color='indianred')
        ax1.set_xlabel("Default Rate (%)")
        ax1.invert_yaxis()
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    st.divider()
    st.subheader("What Drives Predictions? (SHAP Summary)")
    st.caption("Each dot is one applicant from a sample of the data. Features are ranked top to bottom by "
               "overall importance; red = high feature value, blue = low feature value; "
               "position right of center = pushes toward Default.")

    with st.spinner("Loading feature importance summary..."):
        shap_vals_global, shap_sample = load_global_shap()
        shap.summary_plot(shap_vals_global, shap_sample, show=False)
        fig3 = plt.gcf()
        st.pyplot(fig3)
        plt.close(fig3)

    st.divider()
    st.subheader("Full Model Comparison Table")
    st.dataframe(comparison_df[['Model', 'Variant', 'Accuracy', 'Precision', 'Recall', 'F1']].style.format({
        'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}'
    }))


# ---------------- Native multi-page navigation ----------------
# Using st.Page/st.navigation instead of a manual st.radio + if/else means Streamlit
# only executes the code for the CURRENTLY SELECTED page on each run. The other page's
# function is never called, so there's no way for Predict-page widgets/results to
# render on the Model Insights page (or vice versa) — this also removes the page-switch
# "flash" that a single-script if/else layout has.
pg = st.navigation([
    st.Page(predict_page, title="Predict", url_path="predict"),
    st.Page(insights_page, title="Model Insights", url_path="insights"),
])
pg.run()