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

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Sentiment Analyzer</title>

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
            background: rgba(255, 255, 255, 0.97);
            padding: 40px;
            border-radius: 25px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
            text-align: center;
        }

        .icon {
            font-size: 55px;
            margin-bottom: 10px;
        }

        h1 {
            color: #333;
            font-size: 36px;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #777;
            font-size: 16px;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            height: 160px;
            padding: 18px;
            border: 2px solid #ddd;
            border-radius: 15px;
            resize: none;
            font-size: 16px;
            outline: none;
            transition: 0.3s;
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
            transition: 0.3s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 18px;
            background: #f5f5f5;
        }

        .result h2 {
            color: #444;
            margin-bottom: 15px;
        }

        .sentiment {
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .confidence {
            color: #666;
            font-size: 16px;
        }

        .positive {
            color: #159957;
        }

        .negative {
            color: #d63031;
        }

        .footer {
            margin-top: 25px;
            color: #999;
            font-size: 13px;
        }

        @media (max-width: 600px) {
            .container {
                padding: 25px;
            }

            h1 {
                font-size: 28px;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <div class="icon">💬</div>

    <h1>Sentiment Analyzer</h1>

    <p class="subtitle">
        Enter a review and let the AI predict its sentiment.
    </p>

    <form method="POST">

        <textarea
            name="review"
            placeholder="Write your review here..."
            required>{{ review }}</textarea>

        <button type="submit">
            🔍 Analyze Sentiment
        </button>

    </form>

    {% if sentiment %}

    <div class="result">

        <h2>Prediction Result</h2>

        {% if "Positive" in sentiment %}
            <div class="sentiment positive">
                {{ sentiment }}
            </div>
        {% elif "Negative" in sentiment %}
            <div class="sentiment negative">
                {{ sentiment }}
            </div>
        {% else %}
            <div class="sentiment">
                {{ sentiment }}
            </div>
        {% endif %}

        {% if confidence %}
            <p class="confidence">
                Model Confidence: <strong>{{ confidence }}%</strong>
            </p>
        {% endif %}

    </div>

    {% endif %}

    <div class="footer">
        Powered by TF-IDF + Multinomial Naive Bayes
    </div>

</div>

</body>
</html>
```

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
```
