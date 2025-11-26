from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":

        # Fetch inputs
        holiday = int(request.form.get("holiday"))      # already encoded
        temp = float(request.form.get("temp"))
        rain = int(request.form.get("rain"))
        snow = int(request.form.get("snow"))
        weather = int(request.form.get("weather"))      # already encoded
        hours = int(request.form.get("hours"))
        minutes = int(request.form.get("minutes"))
        seconds = int(request.form.get("seconds"))

        # Build final feature list IN CORRECT ORDER (8 features)
        final_features = np.array([
            temp, rain, snow,
            hours, minutes, seconds,
            holiday, weather
        ]).reshape(1, -1)

        prediction = int(model.predict(final_features)[0])

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
