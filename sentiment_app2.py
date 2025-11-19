from flask import Flask, render_template, request, send_file, url_for
import pandas as pd
from transformers import pipeline
from collections import Counter
import json
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")


label_map = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive',
    'NEGATIVE': 'Negative',
    'NEUTRAL': 'Neutral',
    'POSITIVE': 'Positive'
}

def get_sentiment(text):
    result = classifier(text[:512])[0]['label']
    return label_map.get(result, "Unknown")

def generate_wordcloud(text_list, output_file):
    text = ' '.join(text_list)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    wc.to_file(output_file)

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = []
    table_data = None
    chart_data = {}
    metrics = {}

    if request.method == "POST":
        if "feedback" in request.form:
            feedback = request.form.get("feedback")
            sentiment = get_sentiment(feedback)
            predictions.append((feedback, sentiment))

        elif "file" in request.files:
            file = request.files["file"]
            if file and file.filename.endswith(".csv"):
                df = pd.read_csv(file)
                if "text" in df.columns:
                    df['sentiment'] = df['text'].apply(lambda x: get_sentiment(str(x)))
                    table_data = df[['text', 'sentiment']].values.tolist()

                    
                    chart_data = json.dumps(dict(Counter(df['sentiment'])))

                    metrics = dict(Counter(df['sentiment']))

                    os.makedirs("static", exist_ok=True)

                    df.to_csv("static/predicted_sentiment.csv", index=False)

                    generate_wordcloud(df['text'].astype(str).tolist(), "static/wordcloud.png")

                else:
                    predictions.append(("❌ CSV must contain a 'text' column", "Error"))

    return render_template("index.html",
                           predictions=predictions,
                           table_data=table_data,
                           chart_data=chart_data,
                           metrics=metrics)

if __name__ == "__main__":
    app.run(debug=True)
