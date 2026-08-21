import streamlit as st
import textwrap
import joblib
import json
import re
import pandas as pd

def html_md(raw: str):
    """
    Render an HTML string safely, bypassing st.markdown entirely.
    st.markdown(..., unsafe_allow_html=True) still runs its markdown
    parser first -- stray characters like the '*' in CSS attribute
    selectors (e.g. [class*="css"]) get misread as markdown emphasis
    and can corrupt/leak raw HTML or CSS as visible text. st.html()
    renders the string as literal HTML with no markdown parsing at
    all, so this can't happen regardless of indentation or content.
    """
    lines = [line.strip() for line in textwrap.dedent(raw).split("\n")]
    st.html("\n".join(lines))


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sorting Desk — Email Spam Classifier",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# LOAD MODEL + ASSETS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("spam_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return model, vectorizer, metrics

model, vectorizer, metrics = load_assets()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------------------------
# STYLE — Postal / Sorting-Desk theme
# ---------------------------------------------------------------------------
html_md("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --paper: #EFE7D8;
    --paper-light: #F6F1E6;
    --ink: #1B2438;
    --ink-soft: #4A5268;
    --brass: #A6802E;
    --brass-light: #C9A857;
    --wax: #A93A32;
    --wax-dark: #7E2A24;
    --sage: #3F6B4F;
    --sage-dark: #2C4D38;
    --line: #C9BEA4;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(ellipse at top left, #F6F1E6 0%, var(--paper) 55%, #E6DBC2 100%);
    color: var(--ink);
}

#MainMenu, footer, header {visibility: hidden;}

/* ---------- Header ---------- */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--brass);
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.masthead {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 3rem;
    line-height: 1.05;
    color: var(--ink);
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.01em;
}
.masthead em { color: var(--wax); font-style: italic; font-weight: 500; }
.subhead {
    font-size: 1.02rem;
    color: var(--ink-soft);
    max-width: 640px;
    line-height: 1.55;
    margin-bottom: 0.5rem;
}
.desk-divider {
    border: none;
    border-top: 1.5px dashed var(--line);
    margin: 1.6rem 0 1.8rem 0;
}

/* ---------- Envelope input card ---------- */
.envelope-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 0.5rem;
    display: block;
}
div[data-testid="stTextArea"] textarea {
    background: var(--paper-light) !important;
    border: 1.5px dashed var(--line) !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.92rem !important;
    color: var(--ink) !important;
    padding: 1.1rem !important;
    line-height: 1.6 !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--brass) !important;
    box-shadow: none !important;
}

div.stButton > button {
    background: var(--ink);
    color: var(--paper-light);
    border: none;
    border-radius: 3px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.03em;
    padding: 0.65rem 1.6rem;
    transition: all 0.15s ease;
    width: 100%;
}
div.stButton > button:hover {
    background: var(--wax-dark);
    color: white;
}

/* ---------- Postmark stamp (the signature element) ---------- */
.stamp-wrap { display:flex; justify-content:center; align-items:center; padding: 1.2rem 0 0.6rem 0; }
.postmark {
    width: 168px; height: 168px;
    border-radius: 50%;
    border: 3.5px solid var(--stamp-color);
    display: flex; flex-direction: column; align-items:center; justify-content:center;
    transform: rotate(-9deg);
    position: relative;
    font-family: 'Fraunces', serif;
    color: var(--stamp-color);
    box-shadow: inset 0 0 0 5px var(--paper), inset 0 0 0 6.5px var(--stamp-color);
    opacity: 0;
    animation: stampDown 0.4s ease-out 0.05s forwards;
}
@keyframes stampDown {
    0% { opacity: 0; transform: rotate(-9deg) scale(1.6); }
    70% { opacity: 1; }
    100% { opacity: 1; transform: rotate(-9deg) scale(1); }
}
.postmark .verdict { font-size: 1.55rem; font-weight: 700; letter-spacing: 0.03em; margin-top: 2px;}
.postmark .sub { font-family:'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.15em; margin-top: 4px; text-transform: uppercase;}
.postmark .conf { font-family:'IBM Plex Mono', monospace; font-size: 0.85rem; margin-top: 6px; font-weight: 600;}

/* ---------- Metric / receipt cards ---------- */
.receipt-card {
    background: var(--paper-light);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    height: 100%;
}
.receipt-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--ink-soft);
}
.receipt-value {
    font-family: 'Fraunces', serif;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--ink);
    margin-top: 0.15rem;
}
.receipt-value.wax { color: var(--wax); }
.receipt-value.sage { color: var(--sage); }
.receipt-value.brass { color: var(--brass); }

.word-chip {
    display:inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 0.28rem 0.6rem;
    margin: 0.2rem 0.3rem 0.2rem 0;
    color: var(--ink-soft);
}

section[data-testid="stSidebar"] {
    background: var(--ink);
}
section[data-testid="stSidebar"] * { color: var(--paper-light) !important; }
section[data-testid="stSidebar"] hr { border-color: #3A4258 !important; }
</style>
""")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.html("<div class='eyebrow' style='color:#C9A857;'>ARCH TECHNOLOGIES</div>")
    st.html("<div style='font-family:Fraunces,serif; font-size:1.4rem; font-weight:700; margin-bottom:1rem;'>ML Internship<br>Month 1 · Task 1</div>")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("TF-IDF + Multinomial Naive Bayes")
    st.markdown("**Training data**")
    st.markdown(f"{metrics['total_size']:,} labeled emails\n\n{metrics['ham_count']:,} legitimate · {metrics['spam_count']:,} spam")
    st.markdown("---")
    st.markdown("**Performance**")
    st.markdown(f"Accuracy — {metrics['accuracy']*100:.1f}%")
    st.markdown(f"Precision — {metrics['precision']*100:.1f}%")
    st.markdown(f"Recall — {metrics['recall']*100:.1f}%")
    st.markdown(f"F1-score — {metrics['f1']*100:.1f}%")
    st.markdown("---")
    st.html("<span style='font-size:0.78rem; opacity:0.75;'>Dataset: 190K+ Spam/Ham Email Dataset (Kaggle) — real email text sourced from the Enron and related corpora.</span>")

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.html("<div class='eyebrow'>TASK 01 — MACHINE LEARNING</div>")
st.html("<div class='masthead'>The Sorting Desk<br><em>an email triage instrument</em></div>")
st.html(
    "<div class='subhead'>Paste any email or message below. The desk reads it the way a spam "
    "filter does — breaking the text into weighted terms and comparing the pattern against "
    "thousands of previously sorted letters — then stamps a verdict.</div>"
)
st.html("<hr class='desk-divider'>")

# ---------------------------------------------------------------------------
# MAIN: INPUT + VERDICT
# ---------------------------------------------------------------------------
col_input, col_verdict = st.columns([1.3, 1], gap="large")

with col_input:
    st.html("<span class='envelope-label'>✉ Contents of the letter</span>")
    email_text = st.text_area(
        "email_input",
        height=260,
        placeholder="Paste the email or message text here...\n\ne.g. \"Congratulations! You've WON a $1000 Walmart gift card. Click here to claim your prize now!!!\"",
        label_visibility="collapsed",
    )
    analyze = st.button("Analyze Email", use_container_width=True)

    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        if st.button("Try a spam example", use_container_width=True):
            st.session_state["sample"] = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
            st.rerun()
    with sample_col2:
        if st.button("Try a normal example", use_container_width=True):
            st.session_state["sample"] = "Hey, are we still on for the meeting tomorrow at 10am? Let me know if you need to reschedule."
            st.rerun()

if "sample" in st.session_state and not email_text:
    email_text = st.session_state["sample"]

with col_verdict:
    st.html("<span class='envelope-label'>◎ Verdict</span>")
    if analyze and email_text.strip():
        cleaned = clean_text(email_text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = proba[pred] * 100
        is_spam = pred == 1

        stamp_color = "var(--wax)" if is_spam else "var(--sage)"
        verdict_text = "SPAM" if is_spam else "SAFE"
        sub_text = "flagged &amp; sorted" if is_spam else "cleared for delivery"

        html_md(f"""
        <div class='stamp-wrap'>
            <div class='postmark' style='--stamp-color:{stamp_color};'>
                <div class='verdict'>{verdict_text}</div>
                <div class='sub'>{sub_text}</div>
                <div class='conf'>{confidence:.1f}% confident</div>
            </div>
        </div>
        """)

        # top contributing words
        feature_names = vectorizer.get_feature_names_out()
        log_prob_spam = model.feature_log_prob_[1]
        present_idx = vec.nonzero()[1]
        if len(present_idx) > 0:
            word_scores = sorted(
                [(feature_names[i], log_prob_spam[i]) for i in present_idx],
                key=lambda x: x[1], reverse=True
            )[:6]
            st.html("<div style='margin-top:1rem; font-family:IBM Plex Mono, monospace; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--ink-soft);'>Words that influenced this reading</div>")
            chips = "".join([f"<span class='word-chip'>{w}</span>" for w, _ in word_scores])
            st.html(f"<div style='margin-top:0.4rem;'>{chips}</div>")
    else:
        html_md("""
        <div style='background:var(--paper-light); border:1.5px dashed var(--line); border-radius:6px;
                    padding:2.4rem 1.5rem; text-align:center; color:var(--ink-soft); font-size:0.9rem;'>
            Paste a message and press <strong>Analyze Email</strong><br>to see it stamped and sorted.
        </div>
        """)

st.html("<hr class='desk-divider'>")

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE — for the report
# ---------------------------------------------------------------------------
st.html("<div class='eyebrow'>MODEL PERFORMANCE</div>")
st.html("<div style='font-family:Fraunces,serif; font-size:1.6rem; font-weight:600; margin-bottom:1rem;'>Evaluated on a held-out test set</div>")

m1, m2, m3, m4 = st.columns(4)
metric_data = [
    (m1, "Accuracy", f"{metrics['accuracy']*100:.1f}%", ""),
    (m2, "Precision", f"{metrics['precision']*100:.1f}%", "wax"),
    (m3, "Recall", f"{metrics['recall']*100:.1f}%", "sage"),
    (m4, "F1-Score", f"{metrics['f1']*100:.1f}%", "brass"),
]
for col, label, value, cls in metric_data:
    with col:
        html_md(f"""
        <div class='receipt-card'>
            <div class='receipt-label'>{label}</div>
            <div class='receipt-value {cls}'>{value}</div>
        </div>
        """)

st.html("<div style='height:1.2rem;'></div>")

col_cm, col_notes = st.columns([1, 1.2], gap="large")
with col_cm:
    st.html("<div class='receipt-label' style='margin-bottom:0.6rem;'>CONFUSION MATRIX</div>")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Ham", "Actual: Spam"],
        columns=["Predicted: Ham", "Predicted: Spam"]
    )
    st.dataframe(cm_df, use_container_width=True)

with col_notes:
    st.html("<div class='receipt-label' style='margin-bottom:0.6rem;'>READING THESE NUMBERS</div>")
    html_md(f"""
    <div style='font-size:0.88rem; line-height:1.7; color:var(--ink-soft);'>
    <strong>Accuracy</strong> — of all {metrics['test_size']} test messages, how many were sorted correctly overall.<br>
    <strong>Precision</strong> — of the messages the model called "spam," how many actually were. High precision means safe emails rarely get wrongly binned.<br>
    <strong>Recall</strong> — of the messages that were actually spam, how many the model caught. <br>
    <strong>F1-score</strong> — the balance between precision and recall in one number.
    </div>
    """)

st.html("<div style='height:2rem;'></div>")
st.html("<div style='text-align:center; font-family:IBM Plex Mono, monospace; font-size:0.7rem; letter-spacing:0.1em; color:var(--ink-soft); opacity:0.7;'>SORTING DESK — BUILT FOR ARCH TECHNOLOGIES ML INTERNSHIP, MONTH 1</div>")
