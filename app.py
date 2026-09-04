from flask import Flask, request, render_template_string
import joblib
import os

app = Flask(__name__)

# --------------------------------------------------
# Load Model and Vectorizer
# --------------------------------------------------

MODEL_PATH = "sentiment.pkl"
VECTORIZER_PATH = "vector.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# --------------------------------------------------
# HTML + CSS
# --------------------------------------------------

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>AI Sentiment Analyzer</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea, #764ba2);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 750px;
            background: white;
            padding: 40px;
            border-radius: 25px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
            text-align: center;
        }

        .logo {
            font-size: 55px;
            margin-bottom: 10px;
        }

        h1 {
            font-size: 36px;
            color: #333;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #777;
            font-size: 16px;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            height: 170px;
            resize: none;
            border: 2px solid #ddd;
            border-radius: 15px;
            padding: 18px;
            font-size: 16px;
            outline: none;
        }

        textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.2);
        }

        button {
            width: 100%;
            margin-top: 20px;
            padding: 16px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            transform: translateY(-2px);
        }

        .clear-button {
            display: inline-block;
            margin-top: 12px;
            color: #667eea;
            text-decoration: none;
            font-size: 14px;
        }

        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 18px;
            background: #f7f7f7;
        }

        .result-title {
            font-size: 20px;
            color: #555;
            margin-bottom: 15px;
        }

        .positive {
            color: #159957;
            font-size: 32px;
            font-weight: bold;
        }

        .negative {
            color: #d63031;
            font-size: 32px;
            font-weight: bold;
        }

        .neutral {
            color: #e67e22;
            font-size: 32px;
            font-weight: bold;
        }

        .confidence {
            margin-top: 12px;
            color: #666;
            font-size: 16px;
        }

        .footer {
            margin-top: 25px;
            color: #999;
            font-size: 13px;
        }

    </style>

</head>

<body>

<div class="container">

    <div class="logo">💬</div>

    <h1>AI Sentiment Analyzer</h1>

    <p class="subtitle">
        Enter a review and let our machine learning model analyze its sentiment.
    </p>

    <form method="POST">

        <textarea
            name="review"
            placeholder="Example: This product is amazing and I really loved it!"
            required>{{ review }}</textarea>

        <button type="submit">
            🔍 Analyze Sentiment
        </button>

    </form>

    {% if sentiment %}

    <div class="result">

        <div class="result-title">
            Prediction Result
        </div>

        {% if sentiment_type == "positive" %}

            <div class="positive">
                😊 {{ sentiment }}
            </div>

        {% elif sentiment_type == "negative" %}

            <div class="negative">
                😞 {{ sentiment }}
            </div>

        {% else %}

            <div class="neutral">
                😐 {{ sentiment }}
            </div>

        {% endif %}

        {% if confidence is not none %}

        <div class="confidence">
            Model Confidence:
            <strong>{{ confidence }}%</strong>
        </div>

        {% endif %}

    </div>

    {% endif %}

    <div class="footer">
        Powered by TF-IDF + Multinomial Naive Bayes
    </div>

</div>

</body>

</html>
"""


# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    sentiment_type = "neutral"
    confidence = None
    review = ""

    if request.method == "POST":

        review = request.form.get("review", "").strip()

        if review:

            # Convert review into TF-IDF vector
            review_vector = vectorizer.transform([review])

            # Prediction
            prediction = model.predict(review_vector)[0]

            # Convert prediction to string
            prediction_text = str(prediction).lower()

            # --------------------------------------------------
            # Detect Sentiment
            # --------------------------------------------------

            if prediction_text in ["1", "positive", "pos"]:

                sentiment = "Positive"
                sentiment_type = "positive"

            elif prediction_text in ["0", "negative", "neg"]:

                sentiment = "Negative"
                sentiment_type = "negative"

            elif prediction_text in ["2", "neutral", "neu"]:

                sentiment = "Neutral"
                sentiment_type = "neutral"

            else:

                sentiment = str(prediction)

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(review_vector)

                confidence = round(
                    float(max(probabilities[0])) * 100,
                    2
                )

    return render_template_string(
        HTML,
        sentiment=sentiment,
        sentiment_type=sentiment_type,
        confidence=confidence,
        review=review
    )


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
