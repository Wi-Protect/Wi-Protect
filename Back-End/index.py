from flask import Flask

from livePrediction import calibrate, changeModeToDay, changeModeToNight, startPredictions, stopPredictions

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/start_prediction', methods=['POST'])
def predict():

    startPredictions()
    return "Prediction started"


@app.route('/stop_prediction', methods=['POST'])
def stop_prediction():
    stopPredictions()
    return "Prediction stopped"


@app.route('/night-mode', methods=['POST'])
def night_mode():
    changeModeToNight()
    return "Night mode activated"


@app.route('/day-mode', methods=['POST'])
def day_mode():
    changeModeToDay()
    return "Day mode activated"


@app.route('/caliberate', methods=['POST'])
def calibrate_model():
    calibrate()
    return "Calibration Complete!"


if __name__ == '__main__':
    app.run(host='192.168.224.46', port=5000)
