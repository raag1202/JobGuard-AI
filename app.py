import os
import joblib
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="JobGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# LOAD MODELS
# =========================================================
LOGISTIC_MODEL_FILE = "jobguard_logistic_pipeline.pkl"
RF_MODEL_FILE = "jobguard_random_forest_pipeline.pkl"


@st.cache_resource
def load_models():
    logistic_model = None
    random_forest_model = None

    if os.path.exists(LOGISTIC_MODEL_FILE):
        logistic_model = joblib.load(LOGISTIC_MODEL_FILE)

    if os.path.exists(RF_MODEL_FILE):
        random_forest_model = joblib.load(RF_MODEL_FILE)

    return logistic_model, random_forest_model


logistic_model, random_forest_model = load_models()


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .fake-box {
        background-color: #ffe5e5;
        border: 2px solid #ff4b4b;
    }

    .real-box {
        background-color: #e5ffe9;
        border: 2px solid #28a745;
    }

    .model-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #777;
        margin-top: 40px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">🛡️ JobGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Fake Job Posting Detection using Machine Learning</div>',
    unsafe_allow_html=True
)


# =========================================================
# INTRODUCTION
# =========================================================
st.write(
    """
    **JobGuard AI** analyzes job advertisements and predicts whether
    a job posting is **Fraudulent** or **Legitimate**.

    The system uses **TF-IDF** for text feature extraction and
    Machine Learning classification models including
    **Logistic Regression** and **Random Forest**.
    """
)


# =========================================================
# MODEL STATUS
# =========================================================
if logistic_model is None:
    st.error(
        "Logistic Regression model file not found. "
        "Please keep jobguard_logistic_pipeline.pkl in the project folder."
    )

if random_forest_model is None:
    st.info(
        "Random Forest model is not available yet. "
        "It will be added after training and evaluation."
    )


# =========================================================
# INPUT SECTION
# =========================================================
st.markdown(
    '<div class="section-title">Enter Job Advertisement Details</div>',
    unsafe_allow_html=True
)

job_title = st.text_input(
    "Job Title",
    placeholder="Example: Software Developer"
)

company_profile = st.text_area(
    "Company Profile",
    placeholder="Enter the company information..."
)

job_description = st.text_area(
    "Job Description",
    placeholder="Enter the complete job description..."
)


# =========================================================
# ANALYSIS FUNCTION
# =========================================================
def predict_model(model, text):
    prediction = model.predict([text])[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba([text])[0][1]

    return int(prediction), probability


# =========================================================
# ANALYZE BUTTON
# =========================================================
if st.button("🔍 Analyze Job Posting", use_container_width=True):

    if not job_title.strip():
        st.warning("Please enter the Job Title.")

    elif not job_description.strip():
        st.warning("Please enter the Job Description.")

    elif not company_profile.strip():
        st.warning("Please enter the Company Profile.")

    elif logistic_model is None:
        st.error("Logistic Regression model is not available.")

    else:

        # -------------------------------------------------
        # Combine the three required project inputs
        # -------------------------------------------------
        combined_text = (
            job_title.strip()
            + " "
            + company_profile.strip()
            + " "
            + job_description.strip()
        )

        st.markdown("---")
        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )

        # =================================================
        # LOGISTIC REGRESSION
        # =================================================
        lr_prediction, lr_probability = predict_model(
            logistic_model,
            combined_text
        )

        lr_probability_percent = (
            lr_probability * 100
            if lr_probability is not None
            else None
        )

        if lr_prediction == 1:

            st.markdown(
                """
                <div class="result-box fake-box">
                    <h2>⚠️ Fraudulent Job Posting</h2>
                    <p>Logistic Regression predicts this posting as potentially fraudulent.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-box real-box">
                    <h2>✅ Legitimate Job Posting</h2>
                    <p>Logistic Regression predicts this posting as legitimate.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        if lr_probability_percent is not None:

            st.write(
                f"**Logistic Regression Fraud Probability:** "
                f"{lr_probability_percent:.2f}%"
            )

        # =================================================
        # RANDOM FOREST
        # =================================================
        if random_forest_model is not None:

            st.markdown("---")

            rf_prediction, rf_probability = predict_model(
                random_forest_model,
                combined_text
            )

            rf_probability_percent = (
                rf_probability * 100
                if rf_probability is not None
                else None
            )

            st.markdown(
                '<div class="section-title">Random Forest Result</div>',
                unsafe_allow_html=True
            )

            if rf_prediction == 1:

                st.error(
                    "⚠️ Random Forest predicts: Fraudulent Job Posting"
                )

            else:

                st.success(
                    "✅ Random Forest predicts: Legitimate Job Posting"
                )

            if rf_probability_percent is not None:

                st.write(
                    f"**Random Forest Fraud Probability:** "
                    f"{rf_probability_percent:.2f}%"
                )


# =========================================================
# MODEL PERFORMANCE
# =========================================================
st.markdown("---")

st.markdown(
    '<div class="section-title">Model Performance</div>',
    unsafe_allow_html=True
)

st.write(
    "The following results were obtained from the Logistic Regression "
    "model on the test dataset."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        "98.63%"
    )

with col2:
    st.metric(
        "Precision",
        "82.63%"
    )

with col3:
    st.metric(
        "Recall",
        "90.75%"
    )


# =========================================================
# TECHNIQUES USED
# =========================================================
st.markdown("---")

st.markdown(
    '<div class="section-title">Machine Learning Techniques Used</div>',
    unsafe_allow_html=True
)

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown(
        """
        ### 1. TF-IDF

        Converts job advertisement text into numerical
        features that can be used by Machine Learning models.
        """
    )

with tech_col2:
    st.markdown(
        """
        ### 2. Logistic Regression

        A classification algorithm used to predict whether
        a job posting is fraudulent or legitimate.
        """
    )

with tech_col3:
    st.markdown(
        """
        ### 3. Random Forest

        An ensemble classification algorithm that can also
        classify job postings as fraudulent or legitimate.
        """
    )


# =========================================================
# HOW IT WORKS
# =========================================================
st.markdown("---")

st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    **Step 1 — Input**

    The user enters:
    - Job Title
    - Job Description
    - Company Profile

    **Step 2 — Text Processing**

    The job advertisement text is converted into numerical
    features using **TF-IDF**.

    **Step 3 — Classification**

    The processed text is given to:
    - Logistic Regression
    - Random Forest

    **Step 4 — Prediction**

    The models classify the job posting as:
    - **Fraudulent**
    - **Legitimate**

    **Step 5 — Evaluation**

    Model performance is evaluated using:
    - Accuracy
    - Precision
    - Recall
    """
)


# =========================================================
# PROJECT OUTCOME
# =========================================================
st.markdown("---")

st.markdown(
    '<div class="section-title">Project Outcome</div>',
    unsafe_allow_html=True
)

st.write(
    """
    JobGuard AI can help job portals identify potentially fraudulent
    job advertisements and support safer online job searching.
    """
)


# =========================================================
# ABOUT
# =========================================================
st.markdown("---")

st.markdown(
    '<div class="section-title">About the Project</div>',
    unsafe_allow_html=True
)

st.write(
    """
    **Project:** Fake Job Posting Detection

    **Problem Type:** NLP Classification

    **Dataset:** Kaggle Fake Job Postings Dataset

    **Input Features:** Job Title, Job Description, Company Profile

    **Machine Learning Techniques:** TF-IDF, Logistic Regression,
    Random Forest

    **Evaluation Metrics:** Accuracy, Precision, Recall

    **Application:** Detection of potentially fraudulent job advertisements
    """
)


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        JobGuard AI | Fake Job Posting Detection |
        AI Applications Lab | Chandigarh University
    </div>
    """,
    unsafe_allow_html=True
)
