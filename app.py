import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
import os

# Force Transformers to ignore CUDA entirely
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load model ONCE (Streamlit caches it)
@st.cache_resource
def load_classifier():
    try:
        classifier = pipeline(
            "text-classification", 
            model="hemitpatel/political_concept_classifer",
            device=-1,  # Force CPU
            model_kwargs={"low_cpu_mem_usage": True, "torch_dtype": torch.float32}
        )
        return classifier, "💻 CPU"
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        raise e

classifier, device_info = load_classifier()

def classifyPoliticalConcept(text: str):
    return classifier(text)

# Page config
st.set_page_config(page_title="Political Concept Classifier", page_icon="🗳️", layout="wide")

# Sidebar with device info
st.sidebar.title("⚙️ System Info")
st.sidebar.metric("Device", "CPU")
st.sidebar.metric("CUDA Available", "❌ No")

# Title
st.title("🗳️ Political Concept Classifier App")
st.write("Paste political text and classify it into concepts like **economy**, **immigration**, or **healthcare**.")

# Left: Input | Right: Output
col1, col2 = st.columns(2)

with col1:
    st.subheader("✏️ Enter Text to Classify")
    input_text = st.text_area(
        "Political text:",
        placeholder="Paste any political statement, article excerpt, or policy text here...",
        height=250
    )
    classify_btn = st.button("Classify Text", type="primary")

with col2:
    st.subheader("📌 Classification Result")
    if classify_btn:
        if not input_text.strip():
            st.warning("Please enter text before clicking classify.")
        else:
            with st.spinner("Classifying..."):
                result = classifyPoliticalConcept(input_text)
                st.success("Done!")

                if result:
                    top_result = result[0]
                    st.metric(
                        label="Predicted Concept",
                        value=top_result['label'],
                        delta=f"{top_result['score']:.2%} confidence"
                    )
                    st.json(result)

# Extra Sample Tools
st.divider()
st.subheader("📊 Random Example Data (Just for Demo)")
sample_data = pd.DataFrame(np.random.randn(20, 3), columns=['Series 1', 'Series 2', 'Series 3'])
st.line_chart(sample_data)
st.caption("Built with Streamlit 🚀")
