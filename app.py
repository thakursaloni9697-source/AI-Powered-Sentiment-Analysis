"""
AI-Powered Sentiment Analysis System - Streamlit Web Application
Features 5 pages:
1. Home: Overview, tech stack, and architecture
2. Single Text Analysis: Real-time inference with confidence score and token-level explainability
3. Batch Sentiment Analysis: CSV upload, batch inference, interactive charts, and CSV export
4. Model Performance: Holdout test metrics, confusion matrix, ROC curve, and loss curves
5. About: Methodology, transformer distillation details, limitations, and future scope
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

import config
from src.model import load_trained_model
from predict import predict_sentiment, predict_batch

# Set Page Config
st.set_page_config(
    page_title="AI-Powered Sentiment Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .badge-positive {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-negative {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .token-pos {
        background-color: rgba(34, 197, 94, 0.28);
        border-bottom: 2px solid #16a34a;
        padding: 2px 4px;
        margin: 1px;
        border-radius: 4px;
        display: inline-block;
    }
    .token-neg {
        background-color: rgba(239, 68, 68, 0.28);
        border-bottom: 2px solid #dc2626;
        padding: 2px 4px;
        margin: 1px;
        border-radius: 4px;
        display: inline-block;
    }
    .token-neutral {
        padding: 2px 4px;
        margin: 1px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading fine-tuned DistilBERT model into memory...")
def get_cached_model():
    """
    Loads and caches the fine-tuned DistilBERT model and tokenizer.
    Ensures the model is loaded only once and never retrained on app launch.
    """
    if not config.MODEL_SAVE_DIR.exists():
        return None, None
    try:
        return load_trained_model(config.MODEL_SAVE_DIR, device=config.DEVICE)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None


def main():
    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "🏠 Home",
            "🔍 Single Text Analysis",
            "📊 Batch Sentiment Analysis",
            "📈 Model Performance",
            "📖 About & Methodology"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Information**")
    device_label = "🚀 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU Mode"
    st.sidebar.info(f"**Compute Device:** {device_label}\n\n**Base Architecture:** DistilBERT")
    
    # Model status indicator in sidebar
    model_exists = config.MODEL_SAVE_DIR.exists() and (
        (config.MODEL_SAVE_DIR / "model.safetensors").exists() or 
        (config.MODEL_SAVE_DIR / "pytorch_model.bin").exists()
    )
    if model_exists:
        st.sidebar.success("✅ Model weights loaded")
    else:
        st.sidebar.warning("⚠️ Model not trained yet.\nRun `python train.py`")

    # Routing
    if page == "🏠 Home":
        render_home_page()
    elif page == "🔍 Single Text Analysis":
        render_single_text_page(model_exists)
    elif page == "📊 Batch Sentiment Analysis":
        render_batch_page(model_exists)
    elif page == "📈 Model Performance":
        render_performance_page()
    elif page == "📖 About & Methodology":
        render_about_page()


# ---------------------------------------------------------------------------
# PAGE 1: HOME
# ---------------------------------------------------------------------------
def render_home_page():
    st.markdown('<div class="main-header">AI-Powered Sentiment Analysis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production-Ready Natural Language Processing using Fine-Tuned DistilBERT</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📌 Project Overview
        This enterprise-grade NLP application leverages a fine-tuned **DistilBERT** (Distilled Bidirectional Encoder Representations from Transformers) model to classify the sentiment of unstructured customer feedback, product reviews, and social media tweets.

        By retaining **97% of BERT's language comprehension capabilities** while operating with **40% fewer parameters** and **60% faster inference**, this solution provides sub-millisecond sentiment scoring suitable for high-volume customer intelligence pipelines.
        """)

        st.markdown("""
        ### 🎯 Core Objectives
        - **Real-Time Sentiment Classification:** Detect positive and negative customer emotion with calibrated confidence scores.
        - **Explainable AI (XAI):** Uncover the exact words and phrases driving model predictions via gradient-based token attribution.
        - **Scalable Batch Processing:** Analyze multi-thousand row CSV review exports with automated reporting and exportable results.
        - **Robust MLOps Architecture:** Modular codebase featuring stratified train/val/test splits, early stopping, and reproducible training.
        """)

    with col2:
        st.markdown("### 🛠️ Technology Stack")
        tech_items = [
            ("Hugging Face Transformers", "DistilBERT Sequence Classifier"),
            ("PyTorch 2.x", "Deep learning training & autograd"),
            ("DistilBERT Tokenizer", "WordPiece subword tokenization"),
            ("Streamlit", "Interactive multi-page web UI"),
            ("Scikit-Learn", "Stratified splitting & evaluation metrics"),
            ("Plotly & Seaborn", "Interactive data visualizations"),
            ("Pandas & NumPy", "Data wrangling & array operations")
        ]
        for tech, desc in tech_items:
            st.markdown(f"**{tech}**  \n*{desc}*")

    st.markdown("---")
    st.markdown("### 📊 Dataset Information")
    d1, d2, d3 = st.columns(3)
    d1.metric("Task Format", "Binary Classification", "Positive / Negative")
    d2.metric("Target Domains", "Reviews & Tweets", "E-commerce & Social")
    d3.metric("Vocabulary Type", "WordPiece Uncased", "30,522 Subwords")


# ---------------------------------------------------------------------------
# PAGE 2: SINGLE TEXT ANALYSIS
# ---------------------------------------------------------------------------
def render_single_text_page(model_exists: bool):
    st.markdown('<div class="main-header">🔍 Single Text Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify custom customer reviews or tweets with token-level explainability</div>', unsafe_allow_html=True)

    if not model_exists:
        st.warning("⚠️ No fine-tuned model checkpoint found in `models/distilbert-sentiment`. Please run `python train.py` first to generate model weights.")
        return

    model, tokenizer = get_cached_model()
    if model is None:
        st.error("Failed to load model. Check logs.")
        return

    # Sample Preset Inputs
    st.markdown("**Try sample inputs or enter your own review below:**")
    preset_cols = st.columns(3)
    preset_text = None
    if preset_cols[0].button("🌟 Sample Positive Review"):
        preset_text = "The camera quality on this phone is absolutely breathtaking! The battery life easily lasts two days, and customer support was warm and helpful."
    if preset_cols[1].button("⚠️ Sample Negative Review"):
        preset_text = "Worst purchase ever! The device overheated within twenty minutes, the build feels terribly cheap, and customer service refused my refund request."
    if preset_cols[2].button("🔄 Sample Mixed/Constructive Review"):
        preset_text = "The audio quality is quite crisp and pleasant, but the plastic casing feels somewhat flimsy for the expensive price tag."

    user_text = st.text_area(
        "Enter customer review or tweet text:",
        value=preset_text if preset_text else "",
        height=130,
        placeholder="Type or paste customer feedback here..."
    )

    explain_toggle = st.checkbox("🔍 Compute Token Attribution (Explainability)", value=True)

    if st.button("🚀 Analyze Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
            return

        with st.spinner("Analyzing text with fine-tuned DistilBERT..."):
            res = predict_sentiment(
                user_text,
                model=model,
                tokenizer=tokenizer,
                device=config.DEVICE,
                include_explanation=explain_toggle
            )

        if "error" in res:
            st.error(res["error"])
            return

        st.markdown("---")
        # Visual Results Row
        r1, r2, r3 = st.columns([1.2, 1.8, 2.0])

        with r1:
            st.markdown("### Predicted Sentiment")
            if res["label"] == "Positive":
                st.markdown('<div class="badge-positive" style="font-size: 1.5rem;">😊 POSITIVE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-negative" style="font-size: 1.5rem;">😡 NEGATIVE</div>', unsafe_allow_html=True)
            
            st.markdown(f"<br>**Confidence:** `{res['confidence'] * 100:.2f}%`", unsafe_allow_html=True)

        with r2:
            st.markdown("### Confidence Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["confidence"] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Confidence ({res['label']})", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10B981" if res["label"] == "Positive" else "#EF4444"},
                    'steps': [
                        {'range': [0, 50], 'color': "#F3F4F6"},
                        {'range': [50, 80], 'color': "#E5E7EB"},
                        {'range': [80, 100], 'color': "#D1D5DB"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r3:
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame({
                "Sentiment": ["Negative", "Positive"],
                "Probability": [res["probabilities"]["Negative"], res["probabilities"]["Positive"]],
                "Color": ["#EF4444", "#10B981"]
            })
            fig_bar = px.bar(
                prob_df,
                x="Sentiment",
                y="Probability",
                text="Probability",
                color="Sentiment",
                color_discrete_map={"Negative": "#EF4444", "Positive": "#10B981"},
                range_y=[0, 1.1]
            )
            fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig_bar.update_layout(showlegend=False, height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Explainability Section
        if explain_toggle and "attributions" in res and res["attributions"]:
            st.markdown("### 💡 Model Interpretability (Token Attribution)")
            st.markdown("""
            *This visualization highlights words that influenced DistilBERT's sentiment prediction using **embedding-level gradient saliency**.  
            <span class="token-pos">Green</span> indicates positive contribution, while <span class="token-neg">Red</span> indicates negative contribution.*
            """, unsafe_allow_html=True)

            html_tokens = []
            for attr in res["attributions"]:
                score = attr["normalized_score"]
                tok = attr["token"]
                # Clean subwords (##word)
                display_tok = tok.replace("##", "")
                if score > 0.15:
                    html_tokens.append(f'<span class="token-pos" title="Score: +{score:.3f}">{display_tok}</span>')
                elif score < -0.15:
                    html_tokens.append(f'<span class="token-neg" title="Score: {score:.3f}">{display_tok}</span>')
                else:
                    html_tokens.append(f'<span class="token-neutral">{display_tok}</span>')

            st.markdown(
                f'<div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1.2rem; border-radius: 8px; line-height: 2.2; font-size: 1.05rem;">{" ".join(html_tokens)}</div>',
                unsafe_allow_html=True
            )


# ---------------------------------------------------------------------------
# PAGE 3: BATCH SENTIMENT ANALYSIS
# ---------------------------------------------------------------------------
def render_batch_page(model_exists: bool):
    st.markdown('<div class="main-header">📊 Batch Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file containing reviews or tweets to perform high-throughput inference</div>', unsafe_allow_html=True)

    if not model_exists:
        st.warning("⚠️ No fine-tuned model checkpoint found in `models/distilbert-sentiment`. Please run `python train.py` first.")
        return

    model, tokenizer = get_cached_model()
    if model is None:
        st.error("Failed to load model. Check logs.")
        return

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    # Option to load included sample file
    sample_file_path = config.DATA_DIR / "sample_test.csv"
    load_sample = False
    if not uploaded_file and sample_file_path.exists():
        if st.button("📁 Or Load Benchmark Sample File (sample_test.csv)"):
            load_sample = True

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
            return
    elif load_sample and sample_file_path.exists():
        df = pd.read_csv(sample_file_path)

    if df is not None:
        st.write(f"**Loaded Preview ({len(df)} rows):**")
        st.dataframe(df.head(5), use_container_width=True)

        text_columns = [c for c in df.columns if df[c].dtype == "object"]
        if not text_columns:
            st.error("No text columns were found in the uploaded CSV.")
            return

        default_col = "text" if "text" in df.columns else text_columns[0]
        selected_col = st.selectbox("Select the column containing review text:", text_columns, index=text_columns.index(default_col) if default_col in text_columns else 0)

        if st.button("⚡ Run Batch Predictions", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing batch text through DistilBERT...")

            texts = df[selected_col].astype(str).tolist()
            preds = predict_batch(texts, model=model, tokenizer=tokenizer, batch_size=config.EVAL_BATCH_SIZE)

            progress_bar.progress(100)
            status_text.text("Batch processing complete!")

            df["predicted_sentiment"] = [p["label"] for p in preds]
            df["confidence"] = [p["confidence"] for p in preds]
            df["prob_positive"] = [p["prob_positive"] for p in preds]
            df["prob_negative"] = [p["prob_negative"] for p in preds]

            st.markdown("---")
            # Metrics & Distribution Row
            pos_count = int((df["predicted_sentiment"] == "Positive").sum())
            neg_count = int((df["predicted_sentiment"] == "Negative").sum())
            total_count = len(df)
            if total_count == 0:
                st.warning("The selected CSV has no rows to analyze.")
                return

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records Analyzed", f"{total_count:,}")
            c2.metric("Positive Sentiment", f"{pos_count:,} ({pos_count/total_count*100:.1f}%)")
            c3.metric("Negative Sentiment", f"{neg_count:,} ({neg_count/total_count*100:.1f}%)")

            # Chart Row
            ch1, ch2 = st.columns(2)
            with ch1:
                chart_df = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative"],
                    "Count": [pos_count, neg_count]
                })
                fig_pie = px.pie(
                    chart_df,
                    names="Sentiment",
                    values="Count",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444"},
                    hole=0.45,
                    title="Sentiment Distribution Ratio"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with ch2:
                fig_conf = px.histogram(
                    df,
                    x="confidence",
                    color="predicted_sentiment",
                    color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444"},
                    nbins=20,
                    title="Prediction Confidence Spread"
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            # Results Table Preview
            st.markdown("### 📋 Prediction Results Preview")
            st.dataframe(df[[selected_col, "predicted_sentiment", "confidence", "prob_positive", "prob_negative"]].head(25), use_container_width=True)

            # Download CSV
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Predictions as CSV",
                data=csv_data,
                file_name="sentiment_predictions_distilbert.csv",
                mime="text/csv",
                type="primary"
            )


# ---------------------------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# ---------------------------------------------------------------------------
def render_performance_page():
    st.markdown('<div class="main-header">📈 Model Evaluation & Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation metrics computed on unseen test holdout data</div>', unsafe_allow_html=True)

    metrics_file = config.METRICS_FILE
    if not metrics_file.exists():
        st.warning("⚠️ Metrics have not been generated yet. Please run `python evaluate.py` or `python train.py` to produce test results.")
        return

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
    m2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    m3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    m4.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")

    if metrics.get("roc_auc") is not None:
        st.caption(f"**ROC-AUC Score:** `{metrics['roc_auc']:.4f}`")

    st.markdown("---")

    # Visualizations Row 1: Confusion Matrix & ROC Curve
    st.markdown("### 🔬 Test Set Diagnostic Plots")
    p1, p2 = st.columns(2)

    with p1:
        if config.CONFUSION_MATRIX_PLOT.exists():
            st.image(str(config.CONFUSION_MATRIX_PLOT), caption="Confusion Matrix on Holdout Test Set", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found.")

    with p2:
        if config.ROC_CURVE_PLOT.exists():
            st.image(str(config.ROC_CURVE_PLOT), caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
        else:
            st.info("ROC Curve plot not found.")

    # Visualizations Row 2: Training Progression
    st.markdown("---")
    st.markdown("### 📉 Training Progression Curves")
    t1, t2 = st.columns(2)

    with t1:
        if config.TRAINING_LOSS_PLOT.exists():
            st.image(str(config.TRAINING_LOSS_PLOT), caption="Cross-Entropy Loss vs Epochs", use_container_width=True)
        else:
            st.info("Training loss curve not found.")

    with t2:
        if config.ACCURACY_CURVE_PLOT.exists():
            st.image(str(config.ACCURACY_CURVE_PLOT), caption="Accuracy Progression vs Epochs", use_container_width=True)
        else:
            st.info("Accuracy curve not found.")

    # Classification Report
    st.markdown("---")
    st.markdown("### 📋 Per-Class Classification Report")
    rep_file = config.CLASSIFICATION_REPORT_FILE
    if rep_file.exists():
        with open(rep_file, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")


# ---------------------------------------------------------------------------
# PAGE 5: ABOUT
# ---------------------------------------------------------------------------
def render_about_page():
    st.markdown('<div class="main-header">📖 About & Project Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep dive into model architecture, distillation, and production design</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏛️ DistilBERT Architecture & Knowledge Distillation
    **DistilBERT** is a lightweight Transformer model trained via **knowledge distillation** on BERT (`bert-base-uncased`). 
    Knowledge distillation is a compression technique where a compact student model is trained to reproduce the behavior of a larger teacher network.

    #### Key Structural Comparison:
    | Architectural Feature | BERT-Base | DistilBERT | Improvement |
    | :--- | :--- | :--- | :--- |
    | **Transformer Layers** | 12 | 6 | 50% Reduction |
    | **Attention Heads** | 12 | 12 | Preserved Attention Capacity |
    | **Hidden Dimension** | 768 | 768 | Equal Representation Width |
    | **Total Parameters** | 110 Million | 66 Million | 40% Parameter Reduction |
    | **Inference Latency** | Baseline (1.0x) | 1.6x - 2.0x faster | 60% Latency Speedup |
    | **Task Performance** | 100% Benchmark | ~97% Retention | Retains language reasoning |

    ---

    ### ⚙️ Training Methodology
    1. **Pretrained Initialization:** We instantiate `distilbert-base-uncased` with pretrained transformer weights and attach a linear classification head (`nn.Linear(768, 2)`).
    2. **Stratified Data Splitting:** The dataset is partitioned into 80% Train, 10% Validation, and 10% Test subsets, maintaining balanced positive and negative class ratios.
    3. **Optimization Strategy:**
       - **Optimizer:** `AdamW` with decoupled weight decay (`0.01`) applied to weight tensors (excluding bias and LayerNorm).
       - **Learning Rate Schedule:** Linear warm-up across the first 10% of total training steps, followed by linear decay.
       - **Regularization:** Early stopping monitor tracking validation loss to prevent overfitting.
       - **Gradient Clipping:** Max gradient norm capped at `1.0` to avoid exploding gradients.

    ---

    ### ⚠️ System Limitations
    - **Context Window:** DistilBERT possesses a maximum sequence length of 512 tokens. Longer reviews are truncated.
    - **Sarcasm & Subtlety:** Highly sarcastic reviews ("Oh great, another broken cord") may challenge sentiment classification heads lacking long-range pragmatic context.
    - **Domain Specificity:** Pretrained on general English texts (Wikipedia + BookCorpus); niche medical or legal sentiment may exhibit slight domain transfer degradation.

    ---

    ### 🚀 Future Improvements & Scaling
    - **Parameter-Efficient Fine-Tuning (PEFT / LoRA):** Train low-rank adapter matrices to reduce trainable weights to <1% for rapid multi-tenant deployment.
    - **Quantization & ONNX Runtime:** Export the model to INT8 ONNX or TensorRT for sub-5ms CPU edge deployment.
    - **Active Learning Feedback Loop:** Capture low-confidence user reviews directly from Streamlit for human-in-the-loop annotation and continual model retraining.
    """)


if __name__ == "__main__":
    main()
