from pathlib import Path
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic_insurance_fraud_dashboard_data.csv"

# Accept either spelling in case the saved model file has the earlier typo.
MODEL_PATH_CANDIDATES = [
    BASE_DIR / "synthetic_insurance_fraud_model.pkl",
    BASE_DIR / "sythetic_insurance_fraud_model.pkl",
]
MODEL_PATH = next((path for path in MODEL_PATH_CANDIDATES if path.exists()), None)


@st.cache_data

def load_data(path_string):
    return pd.read_csv(path_string)


@st.cache_resource

def load_model(path_string):
    return joblib.load(path_string)


try:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH.name}")
    if MODEL_PATH is None:
        expected_names = ", ".join(path.name for path in MODEL_PATH_CANDIDATES)
        raise FileNotFoundError(
            "Model file not found. Expected one of these files in the app.py folder: "
            + expected_names
        )

    data = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))

except Exception as error:
    st.error(f"Could not load the dataset or model: {error}")
    st.stop()

if "fraud_label" not in data.columns:
    st.error("The dataset does not contain the fraud_label target column.")
    st.stop()


# --------------------------------------------------
# 3. DASHBOARD HEADER
# --------------------------------------------------
st.title("🚗 Insurance Fraud Detection Dashboard")
st.write(
    "An interactive dashboard for exploring insurance claims "
    "and examining machine learning predictions."
)
st.info(
    "Academic research prototype. A fraud prediction is a screening result, "
    "not proof of wrongdoing."
)


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section:",
    [
        "Dashboard Overview",
        "Fraud Analysis",
        "Accident Analysis",
        "Policy Analysis",
        "Age Analysis",
        "Model Information",
        "Test a Claim",
    ],
)


# --------------------------------------------------
# 5. DASHBOARD OVERVIEW
# --------------------------------------------------
if section == "Dashboard Overview":
    total_claims = len(data)
    target_numeric = pd.to_numeric(data["fraud_label"], errors="coerce")
    fraudulent_claims = int((target_numeric == 1).sum())
    legitimate_claims = int((target_numeric == 0).sum())
    fraud_rate = fraudulent_claims / total_claims * 100 if total_claims else 0

    st.header("Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Claims", f"{total_claims:,}")
    col2.metric("Fraudulent Claims", f"{fraudulent_claims:,}")
    col3.metric("Legitimate Claims", f"{legitimate_claims:,}")
    col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.subheader("Fraud Distribution")
    counts = target_numeric.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=["Legitimate", "Fraudulent"],
        y=[counts.get(0, 0), counts.get(1, 0)],
        ax=ax,
    )
    ax.set_xlabel("Claim Status")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Legitimate vs Fraudulent Claims")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)


# --------------------------------------------------
# 6. FRAUD ANALYSIS
# --------------------------------------------------
elif section == "Fraud Analysis":
    st.header("Fraud Analysis")
    target_numeric = pd.to_numeric(data["fraud_label"], errors="coerce")
    fraud_counts = target_numeric.value_counts().reindex([0, 1], fill_value=0)
    fraud_table = pd.DataFrame(
        {
            "Claim Status": ["Legitimate", "Fraudulent"],
            "Number of Claims": fraud_counts.values,
        }
    )
    fraud_table["Percentage"] = (
        fraud_table["Number of Claims"] / len(data) * 100 if len(data) else 0
    )
    fraud_table["Percentage"] = fraud_table["Percentage"].round(2)
    st.dataframe(fraud_table, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=fraud_table, x="Claim Status", y="Percentage", ax=ax)
    ax.set_title("Percentage of Claims by Status")
    ax.set_ylabel("Percentage (%)")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 7. ACCIDENT ANALYSIS
# --------------------------------------------------
elif section == "Accident Analysis":
    st.header("Fraud by Accident Type")
    if "accident_type" not in data.columns:
        st.warning("accident_type is not available in the dataset.")
    else:
        working_data = data.copy()
        working_data["fraud_label"] = pd.to_numeric(
            working_data["fraud_label"], errors="coerce"
        )
        accident_table = (
            working_data.groupby("accident_type")["fraud_label"]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )
        accident_table.columns = [
            "Accident Type", "Total Claims", "Fraudulent Claims", "Fraud Rate"
        ]
        accident_table["Fraud Rate (%)"] = (accident_table["Fraud Rate"] * 100).round(2)
        st.dataframe(accident_table, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=accident_table, x="Accident Type", y="Fraud Rate (%)", ax=ax)
        ax.set_title("Fraud Rate by Accident Type")
        ax.set_ylabel("Fraud Rate (%)")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------
# 8. POLICY ANALYSIS
# --------------------------------------------------
elif section == "Policy Analysis":
    st.header("Fraud by Insurance Policy Type")
    if "policy_type" not in data.columns:
        st.warning("policy_type is not available in the dataset.")
    else:
        working_data = data.copy()
        working_data["fraud_label"] = pd.to_numeric(
            working_data["fraud_label"], errors="coerce"
        )
        policy_table = (
            working_data.groupby("policy_type")["fraud_label"]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )
        policy_table.columns = [
            "Policy Type", "Total Claims", "Fraudulent Claims", "Fraud Rate"
        ]
        policy_table["Fraud Rate (%)"] = (policy_table["Fraud Rate"] * 100).round(2)
        policy_table = policy_table.sort_values("Fraud Rate (%)", ascending=False)
        st.dataframe(policy_table, use_container_width=True)

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=policy_table, x="Fraud Rate (%)", y="Policy Type", ax=ax)
        ax.set_title("Fraud Rate by Policy Type")
        ax.set_xlabel("Fraud Rate (%)")
        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------
# 9. AGE ANALYSIS
# --------------------------------------------------
elif section == "Age Analysis":
    st.header("Fraud by Driver Age Group")
    if "driver_age" not in data.columns:
        st.warning("driver_age column is not available in the dataset.")
    else:
        age_data = data.copy()
        age_data["driver_age"] = pd.to_numeric(age_data["driver_age"], errors="coerce")
        age_data["fraud_label"] = pd.to_numeric(age_data["fraud_label"], errors="coerce")
        age_data["AgeGroup"] = pd.cut(
            age_data["driver_age"],
            bins=[0, 25, 35, 45, 55, 120],
            labels=["18-25", "26-35", "36-45", "46-55", "56+"],
            include_lowest=True,
        )
        age_table = (
            age_data.groupby("AgeGroup", observed=False)["fraud_label"]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )
        age_table.columns = [
            "Age Group", "Total Claims", "Fraudulent Claims", "Fraud Rate"
        ]
        age_table["Fraud Rate (%)"] = (age_table["Fraud Rate"] * 100).round(2)
        st.dataframe(age_table, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=age_table, x="Age Group", y="Fraud Rate (%)", ax=ax)
        ax.set_title("Fraud Rate by Driver Age Group")
        ax.set_ylabel("Fraud Rate (%)")
        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------
# 10. MODEL INFORMATION
# --------------------------------------------------
elif section == "Model Information":
    st.header("🤖 Machine Learning Model Configuration")
    st.write("Information about the model loaded by this dashboard:")

    algorithm_name = type(model).__name__
    model_steps = getattr(model, "steps", None)
    if model_steps:
        algorithm_name = type(model_steps[-1][1]).__name__

    st.write(f"**Loaded model type:** {algorithm_name}")
    st.write(f"**Dataset rows:** {len(data):,}")
    st.write(f"**Dataset columns:** {data.shape[1]}")

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        st.write(f"**Features expected by model:** {len(model_features)}")
        with st.expander("View expected feature names"):
            st.write(list(model_features))

    st.warning(
        "Model predictions are screening results, not verified fraud outcomes. "
        "Review performance metrics and investigate flagged claims independently."
    )


# --------------------------------------------------
# 11. TEST A CLAIM AND MANUAL CLAIM ENTRY
# --------------------------------------------------
if section == "Test a Claim":
    st.header("Insurance Claim Testing and Manual Entry")
    st.write(
        "Use **Test a Claim** to get a prediction without saving the submission, "
        "or use **Manual Claim Entry** to evaluate and save a new submission "
        "in a separate CSV file."
    )
    st.info(
        "A fraud prediction is an indication for further review. "
        "It is not proof that fraud has occurred."
    )

    SUBMISSIONS_PATH = BASE_DIR / "manual_claim_submissions.csv"

    # Dataset columns needed to populate the form choices and defaults.
    required_form_columns = [
        "claim_date",
        "county",
        "policy_type",
        "claim_channel",
        "accident_type",
        "driver_age",
        "vehicle_age_years",
        "policy_tenure_months",
        "annual_premium_kes",
        "claim_amount_kes",
        "days_to_report",
        "police_report_filed",
        "witness_available",
        "previous_claims_3yr",
    ]

    missing_form_columns = [
        column for column in required_form_columns if column not in data.columns
    ]
    if missing_form_columns:
        st.error(
            "The dataset is missing columns required by the claim forms: "
            f"{missing_form_columns}. Check the dataset column names."
        )
        st.stop()

    def get_options(column):
        values = data[column].dropna().unique().tolist()
        return sorted(values, key=lambda value: str(value))

    categorical_columns = [
        "county", "policy_type", "claim_channel", "accident_type",
        "police_report_filed", "witness_available", "previous_claims_3yr",
    ]
    empty_categories = [column for column in categorical_columns if not get_options(column)]
    if empty_categories:
        st.error(f"No dropdown values were found for: {empty_categories}")
        st.stop()

    def format_date_like_training_data(selected_date):
        """Format the selected date similarly to values in the CSV."""
        if pd.api.types.is_datetime64_any_dtype(data["claim_date"]):
            return pd.Timestamp(selected_date)

        sample_values = data["claim_date"].dropna()
        if not sample_values.empty:
            sample_text = str(sample_values.iloc[0]).strip()
            formats = [
                "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
                "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
            ]
            for date_format in formats:
                try:
                    datetime.strptime(sample_text, date_format)
                    return selected_date.strftime(date_format)
                except ValueError:
                    continue

        # The earlier project code used this date convention.
        return selected_date.strftime("%m/%d/%Y")

    def prediction_class_is_fraud(class_value):
        return str(class_value).strip().lower() in {
            "1", "1.0", "fraud", "fraudulent", "true"
        }

    def predict_claim(claim_details):
        input_features = pd.DataFrame([claim_details])
        expected_features = getattr(model, "feature_names_in_", None)

        if expected_features is not None:
            expected_features = list(expected_features)
            missing_features = [
                column for column in expected_features
                if column not in input_features.columns
            ]
            if missing_features:
                raise ValueError(
                    "The form is missing model features: "
                    f"{missing_features}. The input form must match the model's "
                    "training features."
                )
            input_features = input_features[expected_features]
        else:
            # This order is used if the saved model does not expose feature_names_in_.
            input_features = input_features[required_form_columns]

        prediction = model.predict(input_features)[0]
        fraud_probability = None
        classes = list(getattr(model, "classes_", []))

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_features)[0]
            fraud_index = next(
                (index for index, value in enumerate(classes)
                 if prediction_class_is_fraud(value)),
                None,
            )
            if fraud_index is not None and fraud_index < len(probabilities):
                fraud_probability = float(probabilities[fraud_index])

        is_fraud_flagged = prediction_class_is_fraud(prediction)
        return is_fraud_flagged, fraud_probability, prediction

    def render_claim_form(form_prefix, save_submission=False):
        if save_submission:
            st.subheader("Manual Claim Entry")
            st.write(
                "Enter a new claim, evaluate it, and save the submission separately "
                "from the historical dataset."
            )
        else:
            st.subheader("Test an Existing or Example Claim")
            st.write(
                "Enter claim details for a prediction. This test does not save a record."
            )

        with st.form(f"{form_prefix}_form"):
            claim_id = ""
            if save_submission:
                claim_id = st.text_input(
                    "Claim ID",
                    value=f"CLM_MANUAL_{np.random.randint(10000, 99999)}",
                    key=f"{form_prefix}_claim_id",
                )

            claim_date_input = st.date_input(
                "Claim Date",
                value=pd.Timestamp.today().date(),
                key=f"{form_prefix}_claim_date",
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Geography and Policy")
                county_input = st.selectbox(
                    "County", get_options("county"), key=f"{form_prefix}_county"
                )
                policy_input = st.selectbox(
                    "Policy Type", get_options("policy_type"), key=f"{form_prefix}_policy_type"
                )
                channel_input = st.selectbox(
                    "Claim Channel", get_options("claim_channel"), key=f"{form_prefix}_claim_channel"
                )
                tenure_input = st.slider(
                    "Policy Tenure (Months)", 0, 150, 24,
                    key=f"{form_prefix}_tenure",
                )

            with col2:
                st.markdown("### Driver and Incident")
                age_input = st.slider(
                    "Driver Age", 18, 100, 35, key=f"{form_prefix}_driver_age"
                )
                vehicle_age_input = st.slider(
                    "Vehicle Age (Years)", 0, 30, 5,
                    key=f"{form_prefix}_vehicle_age",
                )
                accident_input = st.selectbox(
                    "Accident Type", get_options("accident_type"),
                    key=f"{form_prefix}_accident_type",
                )
                report_days_input = st.slider(
                    "Days to Report Accident", 0, 30, 3,
                    key=f"{form_prefix}_report_days",
                )

            with col3:
                st.markdown("### Financial Details and Evidence")
                premium_input = st.number_input(
                    "Annual Premium (KES)", min_value=0, value=45000,
                    key=f"{form_prefix}_premium",
                )
                amount_input = st.number_input(
                    "Claim Amount (KES)", min_value=0, value=120000,
                    key=f"{form_prefix}_claim_amount",
                )
                police_input = st.selectbox(
                    "Police Report Filed?", get_options("police_report_filed"),
                    key=f"{form_prefix}_police_report",
                )
                witness_input = st.selectbox(
                    "Witness Available?", get_options("witness_available"),
                    key=f"{form_prefix}_witness",
                )
                previous_claims_input = st.selectbox(
                    "Previous Claims (Past 3 Years)", get_options("previous_claims_3yr"),
                    key=f"{form_prefix}_previous_claims",
                )

            submit_label = (
                "Submit New Claim, Evaluate and Save"
                if save_submission else "Evaluate Claim"
            )
            submitted = st.form_submit_button(submit_label, use_container_width=True)

        if not submitted:
            return

        if save_submission and not claim_id.strip():
            st.error("Please enter a claim ID.")
            return

        formatted_claim_date = format_date_like_training_data(claim_date_input)
        claim_details = {
            "claim_date": formatted_claim_date,
            "county": county_input,
            "policy_type": policy_input,
            "claim_channel": channel_input,
            "accident_type": accident_input,
            "driver_age": age_input,
            "vehicle_age_years": vehicle_age_input,
            "policy_tenure_months": tenure_input,
            "annual_premium_kes": premium_input,
            "claim_amount_kes": amount_input,
            "days_to_report": report_days_input,
            "police_report_filed": police_input,
            "witness_available": witness_input,
            "previous_claims_3yr": previous_claims_input,
        }

        try:
            is_fraud_flagged, fraud_probability, raw_prediction = predict_claim(claim_details)

            st.divider()
            st.subheader("Prediction Result")
            st.write(f"**Model prediction:** {raw_prediction}")

            if fraud_probability is not None:
                st.metric("Estimated Fraud Probability", f"{fraud_probability * 100:.2f}%")

            if is_fraud_flagged:
                st.warning(
                    "The model flags this claim for further review. "
                    "This prediction does not establish fraud."
                )
                predicted_status = "Flagged for review"
            else:
                st.success(
                    "The model did not flag this claim at its current classification threshold. "
                    "This does not guarantee the claim is legitimate."
                )
                predicted_status = "Not flagged by model"

            st.caption(
                "This is an experimental prediction. Claims should be assessed alongside "
                "other evidence and independent review."
            )

            if save_submission:
                new_record = {
                    "claim_id": claim_id.strip(),
                    "submitted_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                    **claim_details,
                    "raw_model_prediction": str(raw_prediction),
                    "predicted_fraud_status": predicted_status,
                    "predicted_fraud_probability": (
                        fraud_probability if fraud_probability is not None else np.nan
                    ),
                    "review_status": "Pending independent verification",
                }
                new_record_df = pd.DataFrame([new_record])

                if SUBMISSIONS_PATH.exists():
                    previous_submissions = pd.read_csv(SUBMISSIONS_PATH)
                    if (
                        "claim_id" in previous_submissions.columns
                        and claim_id.strip() in previous_submissions["claim_id"].astype(str).values
                    ):
                        st.error(
                            "That Claim ID already exists in the manual submission log. "
                            "Please use a unique Claim ID."
                        )
                        return
                    updated_submissions = pd.concat(
                        [previous_submissions, new_record_df], ignore_index=True
                    )
                else:
                    updated_submissions = new_record_df

                updated_submissions.to_csv(SUBMISSIONS_PATH, index=False)
                st.success(
                    "The new claim has been saved in manual_claim_submissions.csv. "
                    "The historical dataset was not changed."
                )
                st.write(f"**Claim ID:** {claim_id.strip()}")
                st.write("**Review status:** Pending independent verification")
                st.caption(
                    "The stored model prediction is not a confirmed fraud outcome."
                )

        except Exception as error:
            st.error(f"The claim could not be evaluated or recorded. Details: {error}")

    test_tab, manual_tab = st.tabs(["🔎 Test a Claim", "📝 Manual Claim Entry"])
    with test_tab:
        render_claim_form("existing_claim", save_submission=False)
    with manual_tab:
        render_claim_form("manual_claim", save_submission=True)
