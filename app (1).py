import os
from datetime import datetime

import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "niru-nny/SMS_Spam_Detection"


@st.cache_resource(show_spinner="Loading the pretrained NLP model…")
def load_classifier(model_name: str):
    """Download/cache and load a Hugging Face text-classification model."""
    return pipeline("text-classification", model=model_name)


def normalize_label(raw_label: str) -> str:
    """Normalize common spam/ham label variants; retain unknown labels transparently."""
    label = str(raw_label).strip().lower()
    if "spam" in label or label in {"label_1", "1", "positive"}:
        return "Spam"
    if "ham" in label or "not_spam" in label or label in {"label_0", "0", "negative"}:
        return "Ham"
    return str(raw_label)


def classify_message(classifier, message: str):
    result = classifier(message, truncation=True)[0]
    raw_label = result.get("label", "Unknown")
    return {
        "label": normalize_label(raw_label),
        "raw_label": raw_label,
        "score": float(result.get("score", 0.0)),
    }


def show_result(result: dict, message: str):
    is_spam = result["label"].lower() == "spam"
    if is_spam:
        st.error("🚨 Prediction: SPAM")
    elif result["label"].lower() == "ham":
        st.success("✅ Prediction: HAM (likely legitimate)")
    else:
        st.warning(f"Model label: {result['label']}")

    st.metric("Model score", f"{result['score']:.1%}")
    st.progress(min(max(result["score"], 0.0), 1.0))
    st.caption(
        f"Raw model label: `{result['raw_label']}`. "
        "The score is the model's confidence-like output, not a guarantee that a message is safe."
    )


st.title("📩 SMS Spam / Ham Detector")
st.write(
    "Check whether a text message looks like spam using a pretrained "
    "Hugging Face text-classification model."
)

with st.sidebar:
    st.header("Model settings")
    model_name = st.text_input(
        "Hugging Face model ID",
        value=os.getenv("HF_SPAM_MODEL", DEFAULT_MODEL),
        help="The model is downloaded the first time it is used. Internet access is required on first load.",
    )
    st.info("First run requires internet access to download the model. Later runs may use the local Hugging Face cache.")
    st.markdown("---")
    st.caption("Project: NLP Capstone — SMS Spam Classification")

if not model_name.strip():
    st.warning("Enter a valid Hugging Face model ID in the sidebar.")
    st.stop()

try:
    classifier = load_classifier(model_name.strip())
except Exception as exc:
    st.error("Could not load the model. Check the model ID, internet connection, and installed dependencies.")
    st.code(str(exc))
    st.stop()

tab_single, tab_batch, tab_about = st.tabs(["🔎 Single message", "📄 Batch CSV", "ℹ️ About"])

with tab_single:
    st.subheader("Test one SMS")
    message = st.text_area(
        "Enter a message",
        placeholder="Example: Congratulations! You have won a prize. Claim now!",
        height=140,
        key="single_message",
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        check = st.button("Analyze message", type="primary", use_container_width=True)
    with col2:
        st.caption("Try both promotional messages and ordinary personal messages.")

    if check:
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            with st.spinner("Analyzing message…"):
                try:
                    result = classify_message(classifier, message.strip())
                    show_result(result, message.strip())
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.insert(
                        0,
                        {
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Message": message.strip(),
                            "Prediction": result["label"],
                            "Score": round(result["score"], 4),
                            "Raw label": result["raw_label"],
                        },
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    if st.session_state.get("history"):
        st.markdown("#### Recent predictions")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.rerun()

with tab_batch:
    st.subheader("Classify messages from a CSV file")
    st.write("Upload a CSV with a column containing SMS text. The app will add prediction and score columns.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview", data.head())
            text_column = st.selectbox("Choose the message/text column", options=list(data.columns))
            if st.button("Classify CSV", type="primary"):
                messages = data[text_column].fillna("").astype(str).tolist()
                results = []
                progress = st.progress(0)
                status = st.empty()
                for index, text in enumerate(messages):
                    if text.strip():
                        try:
                            prediction = classify_message(classifier, text)
                        except Exception:
                            prediction = {"label": "Error", "raw_label": "Error", "score": 0.0}
                    else:
                        prediction = {"label": "Empty", "raw_label": "Empty", "score": 0.0}
                    results.append(prediction)
                    progress.progress((index + 1) / max(len(messages), 1))
                    status.write(f"Processed {index + 1} of {len(messages)} messages")
                data["Prediction"] = [item["label"] for item in results]
                data["Model score"] = [round(item["score"], 4) for item in results]
                data["Raw model label"] = [item["raw_label"] for item in results]
                st.success("Classification complete.")
                st.dataframe(data, use_container_width=True, hide_index=True)
                csv_bytes = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="spam_ham_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not read or process this CSV: {exc}")

with tab_about:
    st.subheader("How this app works")
    st.markdown(
        """
        1. Loads a text-classification model from Hugging Face.
        2. Sends the entered SMS text to the model.
        3. Displays the predicted label and model score.
        4. Supports batch classification from a CSV file.

        **Important:** The first model load may take time and needs an internet connection.
        Predictions are not guaranteed to be correct. A high score is not proof that a message is safe,
        and a legitimate message can be marked as spam.
        """
    )
    st.warning(
        "Verify the model card's label definitions before relying on the Spam/Ham mapping. "
        "If the model uses custom labels, update `normalize_label()` in `app.py` accordingly."
    )
    st.markdown(f"**Current model:** `{model_name.strip()}`")
