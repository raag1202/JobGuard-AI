
import base64
import io
import re
import os
import shutil
from urllib.parse import urlparse

import joblib
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

# Tesseract: works on Windows and Streamlit Cloud/Linux.
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
# HELPERS
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def image_to_text(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text.strip()


def extract_webpage_text(url):
    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Please enter a valid HTTP/HTTPS URL.")

    headers = {
        "User-Agent": "Mozilla/5.0 (JobGuard-AI/1.0)"
    }

    response = requests.get(
        url.strip(),
        headers=headers,
        timeout=12,
        allow_redirects=True
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        raise ValueError("This link does not appear to contain an HTML webpage.")

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body = soup.get_text(" ", strip=True)

    # Keep enough text for the model while avoiding very large pages.
    text = f"{title} {body}"
    text = re.sub(r"\s+", " ", text).strip()

    return text[:100000]


def find_risk_indicators(text):
    """Detect broader, explainable job-scam signals."""
    text_lower = re.sub(r"\s+", " ", text.lower()).strip()

    rules = {
        "Payment or registration fee": [
            r"\bregistration fee\b",
            r"\bjoining fee\b",
            r"\bprocessing fee\b",
            r"\bapplication fee\b",
            r"\bpay (money|a fee|first|upfront)\b",
            r"\bupfront (payment|fee|cost)\b",
            r"\bsecurity deposit\b",
            r"\bdeposit (money|amount)\b",
            r"\bpayment required\b",
            r"\bpay to (get|secure|receive) (the )?job\b",
            r"\bpay.*before.*job\b",
        ],

        "Urgent hiring language": [
            r"\burgent hiring\b",
            r"\bhiring immediately\b",
            r"\bjoin immediately\b",
            r"\bimmediate joining\b",
            r"\bapply urgently\b",
            r"\bapply today\b",
            r"\bact now\b",
            r"\blimited vacancies\b",
            r"\blimited seats\b",
            r"\bonly \d+ (positions|vacancies|seats)\b",
            r"\blast chance\b",
        ],

        "No-interview claims": [
            r"\bno interview\b",
            r"\bwithout interview\b",
            r"\bno interview required\b",
            r"\bselected without interview\b",
            r"\binterview not required\b",
            r"\bno questions asked\b",
        ],

        "Unrealistic earning claims": [
            r"\bearn huge\b",
            r"\bearn high income\b",
            r"\beasy money\b",
            r"\bquick money\b",
            r"\bquick cash\b",
            r"\bmake money fast\b",
            r"\bguaranteed income\b",
            r"\bguaranteed earnings\b",
            r"\bguaranteed salary\b",
            r"\bearn ₹[\d,]+",
            r"\bearn \$[\d,]+",
            r"\b₹[\d,]+\s*(per month|monthly|per week|weekly)\b",
            r"\$[\d,]+\s*(per month|monthly|per week|weekly)\b",
        ],

        "Personal-information requests": [
            r"\bsend your otp\b",
            r"\bshare otp\b",
            r"\botp.*verification\b",
            r"\bbank details\b",
            r"\bbank account\b",
            r"\baccount number\b",
            r"\bcard details\b",
            r"\bcredit card\b",
            r"\bdebit card\b",
            r"\bupi pin\b",
            r"\bpassword\b",
            r"\baadhaar\b",
            r"\bpan card\b",
            r"\bpersonal details\b",
            r"\bidentity documents\b",
            r"\bgovernment id\b",
            r"\bsend your .*details\b",
        ],

        "Work-from-home pressure": [
            r"\bwork from home\b",
            r"\bwfh\b",
            r"\bhome based job\b",
            r"\bhome job\b",
            r"\bearn from home\b",
        ],

        "WhatsApp or Telegram recruitment": [
            r"\bcontact (us|recruiter|hr) on whatsapp\b",
            r"\bmessage (us|recruiter|hr) on whatsapp\b",
            r"\bwhatsapp (us|me|hr|recruiter)\b",
            r"\bwhatsapp number\b",
            r"\bcontact.*telegram\b",
            r"\bmessage.*telegram\b",
            r"\btelegram (us|me|hr|recruiter)\b",
            r"\bjoin.*telegram\b",
        ],

        "Suspicious recruiter contact": [
            r"\bcontact.*gmail\.com\b",
            r"\brecruiter.*gmail\.com\b",
            r"\bhr.*gmail\.com\b",
            r"\bcontact.*outlook\.com\b",
            r"\bcontact.*yahoo\.com\b",
            r"\bpay.*to.*recruiter\b",
        ],

        "Crypto or unusual payment request": [
            r"\bpay.*bitcoin\b",
            r"\bpay.*crypto\b",
            r"\bcrypto.*payment\b",
            r"\bbitcoin.*payment\b",
            r"\bgift card\b",
            r"\bprepaid card\b",
            r"\btransfer.*wallet\b",
        ],

        "Fake check or money transfer request": [
            r"\bdeposit this check\b",
            r"\bcash this check\b",
            r"\bdeposit.*check.*equipment\b",
            r"\bsend.*money.*back\b",
            r"\btransfer.*remaining amount\b",
            r"\bsend.*difference\b",
        ],

        "Reshipping or package forwarding job": [
            r"\breshipping\b",
            r"\bre-?shipping\b",
            r"\bpackage forwarding\b",
            r"\bforward packages\b",
            r"\breceive packages.*home\b",
            r"\bship packages.*from home\b",
        ],

        "Paid ratings or reviews": [
            r"\bpay.*for.*reviews\b",
            r"\bpaid reviews\b",
            r"\bget paid.*review\b",
            r"\bget paid.*rating\b",
            r"\bboost.*rating.*paid\b",
        ],
    }

    found = []

    for label, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found.append(label)
                break

    return found


# Different indicators have different severity. Payment and sensitive-data
# requests are stronger evidence than ordinary work-from-home wording.
INDICATOR_WEIGHTS = {
    "Payment or registration fee": 1.00,
    "Urgent hiring language": 0.45,
    "No-interview claims": 0.70,
    "Unrealistic earning claims": 0.85,
    "Personal-information requests": 1.00,
    "Work-from-home pressure": 0.25,

    "WhatsApp or Telegram recruitment": 0.65,
    "Suspicious recruiter contact": 0.55,
    "Crypto or unusual payment request": 1.00,
    "Fake check or money transfer request": 1.00,
    "Reshipping or package forwarding job": 0.90,
    "Paid ratings or reviews": 0.75,
}


def calculate_rule_score(indicators):
    """Return a normalized 0-1 scam-evidence score."""
    if not indicators:
        return 0.0

    total_weight = sum(INDICATOR_WEIGHTS.values())
    found_weight = sum(INDICATOR_WEIGHTS.get(item, 0.0) for item in indicators)
    return min(found_weight / total_weight, 1.0)


def analyze_text(text):
    """Run ML prediction plus explainable rule-based risk analysis."""
    cleaned = re.sub(r"\s+", " ", text).strip()

    if not cleaned:
        return None

    prediction = int(model.predict([cleaned])[0])
    ml_risk = float(model.predict_proba([cleaned])[0][1])
    indicators = find_risk_indicators(cleaned)
    rule_score = calculate_rule_score(indicators)

    # Hybrid score: ML provides learned language patterns while rules add
    # transparent scam evidence that users can understand.
    final_risk = min((ml_risk * 0.65) + (rule_score * 0.35), 1.0)

    # Evidence-aware verdicts prevent a single ML score from being treated
    # as absolute proof of fraud.
    if final_risk >= 0.80 and len(indicators) >= 2:
        verdict = "🔴 CRITICAL RISK"
    elif final_risk >= 0.60 or len(indicators) >= 3:
        verdict = "🟠 HIGH RISK"
    elif final_risk >= 0.35 or ml_risk >= 0.45 or indicators:
        verdict = "🟡 NEEDS VERIFICATION"
    else:
        verdict = "🟢 LOW RISK"

    if verdict in ("🔴 CRITICAL RISK", "🟠 HIGH RISK"):
        recommendation = (
            "Do not pay money or share OTP, bank, card, UPI PIN, or other "
            "sensitive information. Verify the employer independently."
        )
    elif verdict == "🟡 NEEDS VERIFICATION":
        recommendation = (
            "Verify the company website, recruiter identity, job source, "
            "and application domain before proceeding."
        )
    else:
        recommendation = (
            "No strong scam evidence was detected. Still verify the employer "
            "and job source before sharing personal information."
        )

    return {
        "prediction": prediction,
        "ml_risk": ml_risk,
        "rule_score": rule_score,
        "final_risk": final_risk,
        "verdict": verdict,
        "indicators": indicators,
        "recommendation": recommendation,
    }


def show_result(result):
    ml_risk = result["ml_risk"]
    rule_score = result["rule_score"]
    final_risk = result["final_risk"]
    verdict = result["verdict"]
    indicators = result["indicators"]
    recommendation = result["recommendation"]

    st.divider()

    if verdict == "🟢 LOW RISK":
        box_class = "safe-box"
        heading = "✅ Low Risk"
        description = "No strong scam evidence was detected by the hybrid analysis."
    elif verdict == "🟡 NEEDS VERIFICATION":
        box_class = "verify-box"
        heading = "🟡 Needs Verification"
        description = "The posting has some uncertainty. Verify the employer before proceeding."
    elif verdict == "🟠 HIGH RISK":
        box_class = "high-box"
        heading = "🟠 High Risk"
        description = "Multiple signals suggest that this job posting may be fraudulent."
    else:
        box_class = "fake-box"
        heading = "🚨 Critical Risk"
        description = "Strong fraud signals were detected. Treat this posting as highly suspicious."

    st.markdown(
        f"""
        <div class="result-box {box_class}">
            <h2>{heading}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Final Risk", f"{final_risk * 100:.2f}%")
    with m2:
        st.metric("ML Risk", f"{ml_risk * 100:.2f}%")
    with m3:
        st.metric("Scam Evidence", f"{rule_score * 100:.2f}%")

    st.progress(min(max(final_risk, 0.0), 1.0))

    if indicators:
        st.subheader("🚩 Suspicious Indicators")
        for item in indicators:
            st.warning(item)
    else:
        st.success("✅ No rule-based suspicious indicators were detected.")

    st.subheader("💡 Recommendation")
    st.info(recommendation)

    with st.expander("🧠 How this result was calculated"):
        st.write(
            "JobGuard combines the trained Logistic Regression model with "
            "transparent scam-indicator rules. The final score uses 65% ML "
            "risk and 35% rule-based evidence."
        )
        st.write(f"**ML risk:** {ml_risk * 100:.2f}%")
        st.write(f"**Rule-based evidence:** {rule_score * 100:.2f}%")
        st.write(f"**Final hybrid risk:** {final_risk * 100:.2f}%")
        st.write(f"**Verdict:** {verdict}")

    st.caption(
        "Risk scores are estimates, not guarantees. A low-risk result does not "
        "prove that an employer is legitimate. Always verify the employer independently."
    )


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }

    .hero {
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(135deg, #111827, #1f2937);
        text-align: center;
        margin-bottom: 22px;
    }

    .hero img {
        width: 310px;
        max-width: 85%;
        border-radius: 14px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.85;
        margin-top: 10px;
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

    .verify-box {
        border: 2px solid #eab308;
        background: rgba(234, 179, 8, 0.08);
    }

    .high-box {
        border: 2px solid #f97316;
        background: rgba(249, 115, 22, 0.08);
    }

    .info-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.3);
        min-height: 145px;
    }

    .footer {
        text-align: center;
        opacity: 0.65;
        padding: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

try:
    with open("jobguard_logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div class="hero">
            <img src="data:image/png;base64,{logo_base64}">
            <p>Fake Job Posting Detection using Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.markdown(
        """
        <div class="hero">
            <h1>🛡️ JobGuard AI</h1>
            <p>Fake Job Posting Detection using Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write(
    "Analyze job advertisements from text, web links, or screenshots "
    "using the trained JobGuard AI model."
)

# --------------------------------------------------
# MODEL PERFORMANCE
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
    "Values are from the trained Logistic Regression model evaluated on the test dataset."
)

st.divider()

# --------------------------------------------------
# SCANNERS
# --------------------------------------------------

st.subheader("🔍 JobGuard Scanners")

tab_text, tab_link, tab_image = st.tabs(
    ["📝 Text Scanner", "🔗 Link Scanner", "📸 Image Scanner"]
)

# ---------------- TEXT SCANNER ----------------

with tab_text:
    st.write("Enter the details of a job advertisement.")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input(
            "Job Title",
            placeholder="Example: Software Engineer",
            key="text_title"
        )

        company_profile = st.text_area(
            "Company Profile",
            placeholder="Enter information about the company...",
            key="text_company"
        )

        requirements = st.text_area(
            "Requirements",
            placeholder="Enter required skills and qualifications...",
            key="text_requirements"
        )

    with col2:
        description = st.text_area(
            "Job Description",
            placeholder="Enter the complete job description...",
            key="text_description"
        )

        benefits = st.text_area(
            "Benefits",
            placeholder="Enter salary, benefits, perks, etc...",
            key="text_benefits"
        )

    if st.button(
        "🔍 Analyze Job Posting",
        width="stretch",
        type="primary",
        key="analyze_text"
    ):
        job_text = " ".join([
            title,
            company_profile,
            description,
            requirements,
            benefits
        ])

        if not job_text.strip():
            st.warning("⚠️ Please enter some job posting information first.")
        else:
            result = analyze_text(job_text)
            show_result(result)

# ---------------- LINK SCANNER ----------------

with tab_link:
    st.write(
        "Paste a public job-posting webpage URL. JobGuard will extract "
        "the visible webpage text and analyze it."
    )

    url = st.text_input(
        "Job Posting URL",
        placeholder="https://example.com/job-posting",
        key="job_url"
    )

    if st.button(
        "🔗 Scan Job Link",
        width="stretch",
        type="primary",
        key="scan_link"
    ):
        if not url.strip():
            st.warning("Please enter a job-posting URL.")
        else:
            try:
                with st.spinner("Reading the job webpage..."):
                    webpage_text = extract_webpage_text(url)

                if len(webpage_text) < 40:
                    st.warning(
                        "Very little readable text was found on this page. "
                        "Try the Text Scanner or upload a screenshot."
                    )
                else:
                    st.success(
                        f"Webpage text extracted successfully ({len(webpage_text):,} characters)."
                    )

                    with st.expander("👀 View extracted text"):
                        st.write(webpage_text[:12000])

                    result = analyze_text(webpage_text)
                    show_result(result)

            except requests.RequestException as exc:
                st.error(
                    "The webpage could not be accessed. Check the URL or "
                    "whether the site blocks automated requests."
                )
            except Exception as exc:
                st.error(f"Could not scan this link: {exc}")

# ---------------- IMAGE SCANNER ----------------

with tab_image:
    st.write(
        "Upload a screenshot or image of a job advertisement. "
        "OCR will extract its text before analysis."
    )

    uploaded_image = st.file_uploader(
        "Upload Job Screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        key="job_image"
    )

    if uploaded_image is not None:
        st.image(
            uploaded_image,
            caption="Uploaded job advertisement",
            width="stretch"
        )

        if st.button(
            "📸 Scan Image",
            width="stretch",
            type="primary",
            key="scan_image"
        ):
            try:
                with st.spinner("Reading text from the image..."):
                    extracted_text = image_to_text(uploaded_image.getvalue())

                if not extracted_text:
                    st.warning(
                        "No readable text was detected. Try a clearer screenshot."
                    )
                else:
                    st.success("Text extracted successfully.")

                    with st.expander("👀 View OCR text"):
                        st.text_area(
                            "Extracted Text",
                            extracted_text,
                            height=220,
                            disabled=True
                        )

                    result = analyze_text(extracted_text)
                    show_result(result)

            except Exception as exc:
                st.error(
                    "Image scanning failed. Make sure the image contains "
                    "clear readable text."
                )
                st.caption(str(exc))

# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.divider()
st.subheader("⚙️ How JobGuard AI Works")

h1, h2, h3 = st.columns(3)

with h1:
    st.markdown(
        """
        <div class="info-card">
            <h3>1️⃣ Input</h3>
            <p>
            Enter job text, paste a webpage link, or upload a job screenshot.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with h2:
    st.markdown(
        """
        <div class="info-card">
            <h3>2️⃣ Text Processing</h3>
            <p>
            JobGuard extracts or receives text and converts it into TF-IDF features.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with h3:
    st.markdown(
        """
        <div class="info-card">
            <h3>3️⃣ AI Prediction</h3>
            <p>
            Logistic Regression predicts the likelihood of a fraudulent posting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# TECHNOLOGY
# --------------------------------------------------

st.divider()
st.subheader("🧠 Technology Used")

tech1, tech2, tech3, tech4, tech5 = st.columns(5)

with tech1:
    st.info("🐍 Python")
with tech2:
    st.info("📊 Scikit-learn")
with tech3:
    st.info("📝 TF-IDF")
with tech4:
    st.info("🤖 Logistic Regression")
with tech5:
    st.info("🔤 OCR")

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()

with st.expander("📚 About This Project"):
    st.write(
        """
        JobGuard AI is a Machine Learning based system designed to detect
        potentially fraudulent job postings.

        The trained model uses TF-IDF feature extraction followed by
        Logistic Regression classification.

        The application supports three input modes:
        1. Direct job-posting text
        2. Public webpage links
        3. Job-advertisement screenshots using OCR

        Dataset:
        Real or Fake: Fake Jobposting Prediction
        """
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🛡️ <b>JobGuard AI</b><br>
        Fake Job Posting Detection | AI Applications Lab<br>
        Chandigarh University
    </div>
    """,
    unsafe_allow_html=True
)
