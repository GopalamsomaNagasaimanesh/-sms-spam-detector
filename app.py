import os
import re
import pickle
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sms_spam_lstm.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "sms_tokenizer.pkl")
MAX_LENGTH = 100

# Load the trained model and the tokenizer saved from the Colab notebook.
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_spam(message):
    if not message or not message.strip():
        return "Please enter an SMS message.", 0.0

    cleaned = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probability = float(model.predict(padded, verbose=0)[0][0])
    result = "🚨 SPAM MESSAGE" if probability >= 0.5 else "✅ HAM (GENUINE MESSAGE)"
    return result, round(probability * 100, 2)

with gr.Blocks(theme=gr.themes.Soft(), title="SMS Spam Detector") as demo:
    gr.Markdown("""
    # 📩 SMS SPAM DETECTOR
    ### Deep Learning Based LSTM Classification System
    Enter an SMS message to check whether it is Spam or Ham.
    """)
    message = gr.Textbox(
        label="Enter Your SMS Message",
        placeholder="Type your message here...",
        lines=4
    )
    check_btn = gr.Button("🔍 Check Message", variant="primary")
    result = gr.Textbox(label="Detection Result")
    probability = gr.Number(label="Spam Probability (%)")
    clear_btn = gr.ClearButton([message, result, probability], value="Clear")
    check_btn.click(detect_spam, inputs=message, outputs=[result, probability])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
