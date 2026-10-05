from flask import Flask, request, jsonify
from flask_cors import CORS

import pickle
import re

from urllib.parse import urlparse


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

CORS(app)


# ==========================================
# LOAD MODEL
# ==========================================

with open(
    "phishing_model.pkl",
    "rb"
) as file:

    model_data = pickle.load(file)


model = model_data["model"]

vectorizer = model_data["vectorizer"]


# ==========================================
# URL FEATURE ANALYSIS
# ==========================================

def analyze_url(url):

    original_url = url


    # Add protocol if missing
    if not re.match(
        r"^[a-zA-Z]+://",
        url
    ):

        url = "http://" + url


    parsed = urlparse(url)

    domain = parsed.netloc

    domain = domain.split("@")[-1]

    domain = domain.split(":")[0]


    # ==========================================
    # URL LENGTH
    # ==========================================

    url_length = len(original_url)


    # ==========================================
    # DOMAIN LENGTH
    # ==========================================

    domain_length = len(domain)


    # ==========================================
    # IP ADDRESS
    # ==========================================

    ip_pattern = (
        r"^(?:\d{1,3}\.){3}\d{1,3}$"
    )

    is_ip = bool(
        re.match(
            ip_pattern,
            domain
        )
    )


    # ==========================================
    # SUBDOMAINS
    # ==========================================

    domain_parts = domain.split(".")

    subdomains = max(
        len(domain_parts) - 2,
        0
    )


    # ==========================================
    # OBFUSCATION
    # ==========================================

    encoded_chars = re.findall(
        r"%[0-9A-Fa-f]{2}",
        original_url
    )


    obfuscation = (
        len(encoded_chars) > 0
        or "@" in original_url
    )


    # ==========================================
    # NUMBERS
    # ==========================================

    numbers = sum(
        char.isdigit()
        for char in original_url
    )


    # ==========================================
    # SPECIAL CHARACTERS
    # ==========================================

    special_characters = sum(
        not char.isalnum()
        for char in original_url
    )


    # ==========================================
    # HTTPS
    # ==========================================

    https = (
        parsed.scheme.lower()
        == "https"
    )


    return {

        "URL Length":
            url_length,

        "Domain Length":
            domain_length,

        "IP Address":
            "Yes" if is_ip else "No",

        "Subdomains":
            subdomains,

        "Obfuscation":
            "Yes" if obfuscation else "No",

        "Numbers":
            numbers,

        "Special Characters":
            special_characters,

        "HTTPS":
            "Yes" if https else "No"

    }


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return jsonify({

        "message":
        "FraudShield ML API is running!"

    })


# ==========================================
# PREDICT
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "error":
                "Request data is missing"

            }), 400


        url = data.get(
            "url",
            ""
        ).strip()


        if not url:

            return jsonify({

                "error":
                "URL is required"

            }), 400


        # ==========================================
        # URL TEXT → TF-IDF
        # ==========================================

        url_vector = vectorizer.transform(
            [url]
        )


        # ==========================================
        # PREDICTION
        # ==========================================

        prediction = int(
            model.predict(
                url_vector
            )[0]
        )


        # ==========================================
        # PROBABILITY
        # ==========================================

        probabilities = model.predict_proba(
            url_vector
        )[0]


        confidence = round(
            max(probabilities) * 100,
            2
        )


        # ==========================================
        # RESULT
        # ==========================================

        if prediction == 0:

            result = "Phishing Website"

            if confidence >= 80:

                risk = "HIGH"

            elif confidence >= 60:

                risk = "MEDIUM"

            else:

                risk = "LOW"


        else:

            result = "Legitimate Website"

            if confidence >= 80:

                risk = "LOW"

            elif confidence >= 60:

                risk = "MEDIUM"

            else:

                risk = "HIGH"


        # ==========================================
        # FEATURE ANALYSIS
        # ==========================================

        features = analyze_url(url)


        # ==========================================
        # RESPONSE
        # ==========================================

        return jsonify({

            "url":
                url,

            "prediction":
                prediction,

            "result":
                result,

            "confidence":
                confidence,

            "risk":
                risk,

            "features":
                features

        })


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )