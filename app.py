import streamlit as st
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="JobGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("jobguard_logistic_pipeline.pkl")


model = load_model()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #1f2937);
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 48px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    opacity: 0.85;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    margin-top: 20px;
    text-align: center;
}

.safe-box {
    border: 2px solid #22c55e;
    background: rgba(34, 197, 94, 0.08);
}

.fake-box {
    border: 2px solid #ef4444;
    background: rgba(239, 68, 68, 0.08);
}

.info-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.3);
    min-height: 150px;
}

.footer {
    text-align: center;
    opacity: 0.65;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🛡️ JobGuard AI</h1>
    <p>Fake Job Posting Detection using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Analyze a job posting using our trained Machine Learning model "
    "and identify patterns commonly associated with fraudulent job advertisements."
)

st.divider()

# --------------------------------------------------
# PROJECT STATS
# --------------------------------------------------

st.subheader("📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Accuracy", "98.63%")

with c2:
    st.metric("Precision", "82.63%")

with c3:
    st.metric("Recall", "90.75%")

with c4:
    st.metric("F1 Score", "86.50%")

st.caption(
    "Performance values are from the trained Logistic Regression model "
    "evaluated on the test dataset."
)

st.divider()

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("🔍 Analyze Job Posting")

col1, col2 = st.columns(2)

with col1:

    title = st.text_input(
        "Job Title",
        placeholder="Example: Software Engineer"
    )

    company_profile = st.text_area(
        "Company Profile",
        placeholder="Enter information about the company..."
    )

    requirements = st.text_area(
        "Requirements",
        placeholder="Enter required skills and qualifications..."
    )

with col2:

    description = st.text_area(
        "Job Description",
        placeholder="Enter the complete job description..."
    )

    benefits = st.text_area(
        "Benefits",
        placeholder="Enter salary, benefits, perks, etc..."
    )

st.write("")

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button(
    "🔍 Analyze Job Posting",
    use_container_width=True,
    type="primary"
):

    job_text = (
        title + " " +
        company_profile + " " +
        description + " " +
        requirements + " " +
        benefits
    )

    if not job_text.strip():

        st.warning(
            "⚠️ Please enter some job posting information first."
        )

    else:

        prediction = model.predict([job_text])[0]

        probability = model.predict_proba([job_text])[0][1]

        fraud_percentage = probability * 100

        st.divider()

        # --------------------------------------------------
        # FAKE RESULT
        # --------------------------------------------------

        if prediction == 1:

            st.markdown("""
            <div class="result-box fake-box">
                <h2>🚨 Potentially Fake Job Posting</h2>
                <p>The model detected patterns associated with fraudulent job postings.</p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                "Fraud Risk Score",
                f"{fraud_percentage:.2f}%"
            )

            st.progress(
                min(probability, 1.0)
            )

            st.warning(
                "⚠️ Be careful before sharing personal information, "
                "paying money, or accepting the offer."
            )

        # --------------------------------------------------
        # LEGITIMATE RESULT
        # --------------------------------------------------

        else:

            st.markdown("""
            <div class="result-box safe-box">
                <h2>✅ Likely Legitimate Job Posting</h2>
                <p>The model did not detect strong fraudulent patterns.</p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                "Fraud Risk Score",
                f"{fraud_percentage:.2f}%"
            )

            st.progress(
                min(probability, 1.0)
            )

            st.success(
                "The posting appears relatively safe according to "
                "the patterns learned by the model."
            )

        st.caption(
            "Note: This is an AI prediction, not a guarantee that a job "
            "posting is genuine or fraudulent."
        )

# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.divider()

st.subheader("⚙️ How JobGuard AI Works")

h1, h2, h3 = st.columns(3)

with h1:
    st.markdown("""
    <div class="info-card">
        <h3>1️⃣ Job Details</h3>
        <p>
        The user enters the title, company profile,
        description, requirements and benefits.
        </p>
    </div>
    """, unsafe_allow_html=True)

with h2:
    st.markdown("""
    <div class="info-card">
        <h3>2️⃣ TF-IDF</h3>
        <p>
        Text is converted into numerical features
        using TF-IDF and n-gram analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

with h3:
    st.markdown("""
    <div class="info-card">
        <h3>3️⃣ Prediction</h3>
        <p>
        Logistic Regression analyzes the features
        and predicts the job posting category.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# TECHNOLOGY
# --------------------------------------------------

st.divider()

st.subheader("🧠 Technology Used")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.info("🐍 Python")

with tech2:
    st.info("📊 Scikit-learn")

with tech3:
    st.info("📝 TF-IDF")

with tech4:
    st.info("🤖 Logistic Regression")

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()

with st.expander("📚 About This Project"):

    st.write("""
    JobGuard AI is a Machine Learning based system designed to detect
    potentially fraudulent job postings.

    The system uses text information from job advertisements and applies
    TF-IDF feature extraction followed by Logistic Regression classification.

    Dataset:
    Real or Fake: Fake Jobposting Prediction

    The model was trained and evaluated using a stratified train-test split.
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    🛡️ <b>JobGuard AI</b><br>
    Fake Job Posting Detection | AI Applications Lab<br>
    Chandigarh University
</div>
""", unsafe_allow_html=True)