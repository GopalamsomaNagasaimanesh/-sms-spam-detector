# SMS Spam Detector — LSTM

A deep-learning SMS classification demo built with TensorFlow/Keras and Gradio.

## Files
- `app.py` — Gradio web application
- `sms_spam_lstm.keras` — trained LSTM model
- `sms_tokenizer.pkl` — fitted Keras tokenizer
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

The app uses the `PORT` environment variable when hosted (for example, Render).

## Model details
The model classifies messages as HAM (0) or SPAM (1), using the same text-cleaning,
tokenization and post-padding process as the training notebook. The decision threshold
is 0.5. Predictions are model estimates and should not be treated as a guarantee that
a message is safe.
