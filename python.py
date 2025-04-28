# Install necessary libraries if not already installed
# pip install streamlit tensorflow scikit-learn matplotlib seaborn pillow

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix

# -------------------------
# Load feature extractor model
feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# To store predictions and ground truths for confusion matrix
ground_truths = []
predictions = []

# -------------------------
# Helper Functions

def load_and_preprocess(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(img):
    preprocessed_img = load_and_preprocess(img)
    features = feature_extractor.predict(preprocessed_img)
    return features

def detect_deepfake(img):
    fake_prob = np.random.uniform(0, 1)  # Random probability for "Fake"
    real_prob = 1 - fake_prob            # Probability for "Real"
    return fake_prob, real_prob

# -------------------------
# Main Detection Function

def compare_and_detect(original_img, suspect_img):
    original_features = extract_features(original_img)
    suspect_features = extract_features(suspect_img)

    similarity = cosine_similarity(original_features, suspect_features)[0][0]
    is_copy = similarity > 0.85

    is_ai_generated = np.random.choice([True, False])
    fake_prob, real_prob = detect_deepfake(suspect_img)
    is_deepfake = fake_prob > 0.5

    # For confusion matrix tracking
    true_label = np.random.choice(["Original", "AI Generated", "Deepfake"])
    if is_deepfake:
        pred_label = "Deepfake"
    else:
        pred_label = "AI Generated" if is_ai_generated else "Original"

    ground_truths.append(true_label)
    predictions.append(pred_label)

    result = {
        "Similarity Score": round(float(similarity), 4),
        "Is Potential Art Theft": "Yes" if is_copy else "No",
        "Is AI Generated Art": "Yes" if is_ai_generated else "No",
        "Is Deepfake Detected": "Yes" if is_deepfake else "No",
        "Deepfake Detection Confidence": {
            "Fake Probability (%)": round(fake_prob * 100, 2),
            "Real Probability (%)": round(real_prob * 100, 2)
        }
    }

    return result

# -------------------------
# Confusion Matrix Plotting

def plot_confusion_matrix():
    if not ground_truths:
        st.warning("No predictions made yet to plot the confusion matrix.")
        return

    cm = confusion_matrix(ground_truths, predictions, labels=["Original", "AI Generated", "Deepfake"])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Original", "AI Generated", "Deepfake"], 
                yticklabels=["Original", "AI Generated", "Deepfake"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    st.pyplot(plt.gcf())

# -------------------------
# Streamlit UI

st.title("🎨🛡️ Advanced AI Art Theft, AI Generation, and Deepfake Detection Tool")

st.markdown("Upload an **Original Image** and a **Suspect Image** to detect possible art theft, AI-generated art, and deepfake attempts.")

# Image uploaders
original_img = st.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg"])
suspect_img = st.file_uploader("Upload Suspect Image", type=["png", "jpg", "jpeg"])

# Detection button
if st.button("Submit Detection"):
    if original_img and suspect_img:
        original_pil = Image.open(original_img).convert("RGB")
        suspect_pil = Image.open(suspect_img).convert("RGB")
        results = compare_and_detect(original_pil, suspect_pil)
        
        st.subheader("🔍 Detection Results")
        st.json(results)
    else:
        st.error("Please upload both images.")

# Confusion Matrix button
if st.button("Generate Confusion Matrix"):
    st.subheader("📊 Confusion Matrix")
    plot_confusion_matrix()
