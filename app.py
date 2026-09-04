```python
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("sentiment.pkl")
vectorizer = joblib.load("vector.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    review = ""
    confidence = None

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if review:
            # Convert review into TF-IDF features
            review_vector = vectorizer.transform([review])

            # Predict sentiment
            prediction = model.predict(review_vector)[0]

            # Get probability/confidence if available
            try:
                probabilities = model.predict_proba(review_vector)[0]
                confidence = round(max(probabilities) * 100, 2)
            except Exception:
                confidence = None

            # Handle different possible label formats
            prediction_str = str(prediction).lower()

            if prediction_str in ["1", "positive", "pos"]:
                sentiment = "Positive 😊"
            elif prediction_str in ["0", "negative", "neg"]:
                sentiment = "Negative 😞"
            else:
                sentiment = str(prediction)

    return render_template(
        "index.html",
        sentiment=sentiment,
        review=review,
        confidence=confidence
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
```
