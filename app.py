import re
import streamlit as st
import joblib
import shap
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

st.set_page_config(page_title="Tamil Fake News Detector", page_icon="📰", layout="centered")

# Authenticity=0 -> Real (routine news reporting), Authenticity=1 -> Fake (viral hoax /
# quote-style misinformation) -- verified against actual headline content in the dataset.
LABELS = {0: "Real", 1: "Fake"}
CLASS_NAMES = ["Real", "Fake"]  # order must match sklearn's classes_ = [0, 1]


def clean_tamil_text(text):
    """Same cleaning function used during training -- must match exactly."""
    text = str(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[a-zA-Z0-9]+', ' ', text)
    text = re.sub(r'[^\u0B80-\u0BFF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_pipeline():
    return joblib.load("tamil_fake_news_pipeline.pkl")


@st.cache_resource
def load_explainers(_pipeline):
    lime_explainer = LimeTextExplainer(class_names=CLASS_NAMES, split_expression=lambda x: x.split())

    def predict_proba_fn(texts):
        return _pipeline.predict_proba(list(texts))

    masker = shap.maskers.Text(tokenizer=r"\s+")
    shap_explainer = shap.Explainer(predict_proba_fn, masker, output_names=CLASS_NAMES)
    shap_explainer(["warm up"])  # pay SHAP's one-time JIT/compile cost here at load time,
                                  # not on the user's first "Detect" click
    return lime_explainer, shap_explainer


pipeline = load_pipeline()
lime_explainer, shap_explainer = load_explainers(pipeline)

st.title("📰 Tamil Fake News Detector")
st.write(
    "Paste a **Tamil news headline** below. The model predicts whether it is likely "
    "**Real** or **Fake**, and LIME/SHAP highlight which words influenced the decision."
)

headline = st.text_area("Tamil headline", height=100,
                         placeholder="இங்கே தமிழ் செய்தி தலைப்பை உள்ளிடவும்...")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    predict_clicked = st.button("🔍 Detect", type="primary", use_container_width=True)
with col2:
    show_lime = st.checkbox("Show LIME explanation", value=True)
with col3:
    show_shap = st.checkbox("Show SHAP explanation", value=True)

if predict_clicked:
    if not headline.strip():
        st.warning("Please enter a Tamil headline first.")
    else:
        cleaned = clean_tamil_text(headline)
        if not cleaned:
            st.error("No Tamil text detected after cleaning -- please enter a Tamil headline.")
        else:
            proba = pipeline.predict_proba([cleaned])[0]
            pred = int(proba.argmax())
            label = LABELS[pred]
            confidence = proba[pred]

            if label == "Real":
                st.success(f"✅ Prediction: **{label}**  (confidence: {confidence:.1%})")
            else:
                st.error(f"⚠️ Prediction: **{label}**  (confidence: {confidence:.1%})")

            st.write(f"P(Real) = {proba[0]:.1%}    |    P(Fake) = {proba[1]:.1%}")

            if show_lime:
                with st.spinner("Generating LIME explanation..."):
                    exp = lime_explainer.explain_instance(
                        cleaned, pipeline.predict_proba, num_features=8)
                    st.subheader("LIME -- why this prediction?")
                    components.html(exp.as_html(), height=400, scrolling=True)

            if show_shap:
                with st.spinner("Generating SHAP explanation..."):
                    shap_values = shap_explainer([cleaned])
                    predicted_class_name = LABELS[pred]
                    html = shap.plots.text(
                        shap_values[0, :, predicted_class_name], display=False)
                    st.subheader("SHAP -- word contribution to prediction")
                    components.html(html, height=250, scrolling=True)

                    st.caption(
                        "Red words push the prediction toward "
                        f"**{predicted_class_name}**; blue words push away from it."
                    )

st.markdown("---")
st.caption(
    "MSc Data Science mini project -- Tamil fake news detection using TF-IDF + "
    "classical ML models, explained with LIME and SHAP."
)
