import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import kagglehub


st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)


@st.cache_data
def load_data():
    """Download and preprocess the same dataset used in the notebook."""
    path = kagglehub.dataset_download("ninzaami/loan-predication")
    csv_files = list(Path(path).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV file was found in the downloaded dataset.")

    df = pd.read_csv(csv_files[0])

    # Match the notebook preprocessing
    df = df.dropna()
    df.replace({"Loan_Status": {"N": 0, "Y": 1}}, inplace=True)
    df = df.replace(to_replace="3+", value=4)
    df.replace(
        {
            "Married": {"No": 0, "Yes": 1},
            "Gender": {"Male": 1, "Female": 0},
            "Self_Employed": {"No": 0, "Yes": 1},
            "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
            "Education": {"Graduate": 1, "Not Graduate": 0},
        },
        inplace=True,
    )

    return df


@st.cache_resource
def train_model():
    """Train the same linear SVM used in the notebook."""
    df = load_data()

    X = df.drop(columns=["Loan_ID", "Loan_Status"], axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        stratify=y,
        random_state=2,
    )

    classifier = svm.SVC(kernel="linear")
    classifier.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))

    return classifier, train_accuracy, test_accuracy


st.title("🏦 Loan Approval Predictor")
st.write("Predict loan approval using the **Linear SVM** model from the project notebook.")

with st.sidebar:
    st.header("About")
    st.write(
        "This app follows the preprocessing and model setup from the original "
        "Loan Approval Prediction notebook."
    )
    st.caption("Model: Support Vector Machine (linear kernel)")
    st.caption("Dataset: Kaggle loan-predication")

try:
    model, train_accuracy, test_accuracy = train_model()

    st.success("Model loaded successfully.")

    with st.expander("Model performance"):
        col1, col2 = st.columns(2)
        col1.metric("Training Accuracy", f"{train_accuracy:.2%}")
        col2.metric("Test Accuracy", f"{test_accuracy:.2%}")

    st.subheader("Enter Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", [0, 1, 2, 4], index=0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=5000.0,
            step=500.0,
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=0.0,
            step=500.0,
        )
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=150.0,
            step=10.0,
            help="Use the same units as the original dataset.",
        )
        loan_term = st.number_input(
            "Loan Amount Term",
            min_value=0.0,
            value=360.0,
            step=12.0,
        )
        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"],
        )

    input_data = pd.DataFrame(
        [
            {
                "Gender": 1 if gender == "Male" else 0,
                "Married": 1 if married == "Yes" else 0,
                "Dependents": dependents,
                "Education": 1 if education == "Graduate" else 0,
                "Self_Employed": 1 if self_employed == "Yes" else 0,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Property_Area": {
                    "Rural": 0,
                    "Semiurban": 1,
                    "Urban": 2,
                }[property_area],
            }
        ]
    )

    if st.button("🔍 Predict Loan Status", type="primary", use_container_width=True):
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
            st.write("The model predicts **Loan_Status = Y**.")
        else:
            st.error("❌ Loan Not Approved")
            st.write("The model predicts **Loan_Status = N**.")

        st.caption(
            "This prediction is produced by the project's SVM model and should not "
            "be treated as a real lending decision."
        )

except Exception as exc:
    st.error("The app could not load the model or dataset.")
    st.code(str(exc))
    st.info(
        "Make sure the deployment environment has internet access and that the "
        "Kaggle dataset can be downloaded."
    )
