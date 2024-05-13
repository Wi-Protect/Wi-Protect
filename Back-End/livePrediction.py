import datetime
import numpy as np
import csv
from io import StringIO
import json
from math import sqrt, atan2
import serial
import joblib
import pywt
import pandas as pd
import time
import threading
import os

from notificationSender import send_notification

DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]

AMP_AND_PHASE_COLUMNS_NAMES = ['id', 'time', 'amplitude1_1', 'amplitude2_1', 'amplitude3_1', 'amplitude4_1', 'amplitude5_1', 'amplitude6_1', 'amplitude7_1', 'amplitude8_1', 'amplitude9_1', 'amplitude10_1', 'amplitude11_1', 'amplitude12_1', 'amplitude13_1', 'amplitude14_1', 'amplitude15_1', 'amplitude16_1', 'amplitude17_1', 'amplitude18_1', 'amplitude19_1', 'amplitude20_1', 'amplitude21_1', 'amplitude22_1', 'amplitude23_1', 'amplitude24_1', 'amplitude25_1', 'amplitude26_1', 'amplitude27_1', 'amplitude28_1', 'amplitude29_1', 'amplitude30_1', 'amplitude31_1', 'amplitude32_1', 'amplitude33_1', 'amplitude34_1', 'amplitude35_1', 'amplitude36_1', 'amplitude37_1', 'amplitude38_1', 'amplitude39_1', 'amplitude40_1', 'amplitude41_1', 'amplitude42_1', 'amplitude43_1', 'amplitude44_1', 'amplitude45_1', 'amplitude46_1', 'amplitude47_1', 'amplitude48_1', 'amplitude49_1', 'amplitude50_1', 'amplitude51_1', 'amplitude52_1', 'amplitude53_1', 'amplitude54_1', 'amplitude55_1', 'amplitude56_1', 'amplitude57_1', 'amplitude58_1', 'amplitude59_1', 'amplitude60_1_1', 'amplitude61_1', 'amplitude62_1', 'amplitude63_1', 'amplitude64_1', 'phase1_1', 'phase2_1', 'phase3_1', 'phase4_1', 'phase5_1', 'phase6_1', 'phase7_1', 'phase8_1', 'phase9_1', 'phase10_1', 'phase11_1', 'phase12_1', 'phase13_1', 'phase14_1', 'phase15_1', 'phase16_1', 'phase17_1', 'phase18_1', 'phase19_1', 'phase20_1', 'phase21_1', 'phase22_1', 'phase23_1', 'phase24_1', 'phase25_1', 'phase26_1', 'phase27_1', 'phase28_1', 'phase29_1', 'phase30_1', 'phase31_1', 'phase32_1', 'phase33_1', 'phase34_1', 'phase35_1', 'phase36_1', 'phase37_1', 'phase38_1', 'phase39_1', 'phase40_1', 'phase41_1', 'phase42_1', 'phase43_1', 'phase44_1', 'phase45_1', 'phase46_1', 'phase47_1', 'phase48_1', 'phase49_1', 'phase50_1', 'phase51_1', 'phase52_1', 'phase53_1', 'phase54_1', 'phase55_1', 'phase56_1', 'phase57_1', 'phase58_1', 'phase59_1', 'phase60_1', 'phase61_1', 'phase62_1', 'phase63_1',
                               'phase64_1', 'amplitude1_2', 'amplitude2_2', 'amplitude3_2', 'amplitude4_2', 'amplitude5_2', 'amplitude6_2', 'amplitude7_2', 'amplitude8_2', 'amplitude9_2', 'amplitude10_2', 'amplitude11_2', 'amplitude12_2', 'amplitude13_2', 'amplitude14_2', 'amplitude15_2', 'amplitude16_2', 'amplitude17_2', 'amplitude18_2', 'amplitude19_2', 'amplitude20_2', 'amplitude21_2', 'amplitude22_2', 'amplitude23_2', 'amplitude24_2', 'amplitude25_2', 'amplitude26_2', 'amplitude27_2', 'amplitude28_2', 'amplitude29_2', 'amplitude30_2', 'amplitude31_2', 'amplitude32_2', 'amplitude33_2', 'amplitude34_2', 'amplitude35_2', 'amplitude36_2', 'amplitude37_2', 'amplitude38_2', 'amplitude39_2', 'amplitude40_2', 'amplitude41_2', 'amplitude42_2', 'amplitude43_2', 'amplitude44_2', 'amplitude45_2', 'amplitude46_2', 'amplitude47_2', 'amplitude48_2', 'amplitude49_2', 'amplitude50_2', 'amplitude51_2', 'amplitude52_2', 'amplitude53_2', 'amplitude54_2', 'amplitude55_2', 'amplitude56_2', 'amplitude57_2', 'amplitude58_2', 'amplitude59_2', 'amplitude60_2', 'amplitude61_2', 'amplitude62_2', 'amplitude63_2', 'amplitude64_2', 'phase1_2', 'phase2_2', 'phase3_2', 'phase4_2', 'phase5_2', 'phase6_2', 'phase7_2', 'phase8_2', 'phase9_2', 'phase10_2', 'phase11_2', 'phase12_2', 'phase13_2', 'phase14_2', 'phase15_2', 'phase16_2', 'phase17_2', 'phase18_2', 'phase19_2', 'phase20_2', 'phase21_2', 'phase22_2', 'phase23_2', 'phase24_2', 'phase25_2', 'phase26_2', 'phase27_2', 'phase28_2', 'phase29_2', 'phase30_2', 'phase31_2', 'phase32_2', 'phase33_2', 'phase34_2', 'phase35_2', 'phase36_2', 'phase37_2', 'phase38_2', 'phase39_2', 'phase40_2', 'phase41_2', 'phase42_2', 'phase43_2', 'phase44_2', 'phase45_2', 'phase46_2', 'phase47_2', 'phase48_2', 'phase49_2', 'phase50_2', 'phase51_2', 'phase52_2', 'phase53_2', 'phase54_2', 'phase55_2', 'phase56_2', 'phase57_2', 'phase58_2', 'phase59_2', 'phase60_2', 'phase61_2', 'phase62_2', 'phase63_2', 'phase64_2', 'rssi_1', 'rssi_2']


save_dir = './models/'

gmm_models = []

for i in range(6):
    model_filename = f'gmm_model_{i}.pkl'
    model_path = save_dir + model_filename
    loaded_gmm_model = joblib.load(model_path)
    gmm_models.append(loaded_gmm_model)

pca = joblib.load('./models/pca_model.joblib')

wavelet_name = 'gaus1'

scales = np.arange(1, 128)

recent = [0, 0, 0, 0, 0]

SERIAL_PORT_1 = '/dev/ttyUSB0'
SERIAL_PORT_2 = '/dev/ttyUSB1'

ser1 = serial.Serial(port=SERIAL_PORT_1, baudrate=921600,
                     bytesize=8, parity='N', stopbits=1)

ser2 = serial.Serial(port=SERIAL_PORT_2, baudrate=921600,
                     bytesize=8, parity='N', stopbits=1)

ifRun = True

NIGHT_TIME = 0
DAY_TIME = 1

mode = NIGHT_TIME

lastQueue = [0, 0, 0, 0, 0, 0, 0]


def csi_data_read_parse():
    global ser1

    global ser2

    count = 0

    csi_list = np.zeros((1, 260))

    startTime = datetime.datetime.now()

    while True:
        strings1 = str(ser1.readline())
        strings2 = str(ser2.readline())
        # print(strings)
        if not strings1 or not strings2:
            continue

        strings1 = strings1.lstrip('b\'').rstrip('\\r\\n\'')
        strings2 = strings2.lstrip('b\'').rstrip('\\r\\n\'')
        index1 = strings1.find('CSI_DATA')
        index2 = strings2.find('CSI_DATA')

        if index1 == -1 or index2 == -1:
            continue

        csv_reader1 = csv.reader(StringIO(strings1))
        csv_reader2 = csv.reader(StringIO(strings2))
        csi_data1 = next(csv_reader1)
        csi_data2 = next(csv_reader2)

        if len(csi_data1) != len(DATA_COLUMNS_NAMES) or len(csi_data2) != len(DATA_COLUMNS_NAMES):
            # print("element number is not equal")
            # print(csi_data1)
            # print(csi_data2)
            continue

        try:
            csi_raw_data1 = json.loads(csi_data1[-1])
            csi_raw_data2 = json.loads(csi_data2[-1])
        except json.JSONDecodeError:
            # print(f"data is incomplete")
            # print(csi_data1)
            # print(csi_data2)
            continue

        if len(csi_raw_data1) != 128 and len(csi_raw_data1) != 256 and len(csi_raw_data1) != 384:
            # print(f"element number is not equal: {len(csi_raw_data1)}")
            # print(csi_data1)
            continue

        if len(csi_raw_data2) != 128 and len(csi_raw_data2) != 256 and len(csi_raw_data2) != 384:
            # print(f"element number is not equal: {len(csi_raw_data2)}")
            # print(csi_data2)
            continue

        imaginary1 = []
        real1 = []
        imaginary2 = []
        real2 = []
        amplitudes = []
        phases = []

        # print(csi_raw_data[0])

        for i in range(len(csi_raw_data1)):
            if i % 2 == 0:
                imaginary1.append(csi_raw_data1[i])
            else:
                real1.append(csi_raw_data1[i])

        for i in range(len(csi_raw_data2)):
            if i % 2 == 0:
                imaginary2.append(csi_raw_data2[i])
            else:
                real2.append(csi_raw_data2[i])

        # Transform imaginary and real into amplitude and phase
        for i in range(int(len(csi_raw_data1) / 2)):
            amplitudes.append(sqrt(imaginary1[i] ** 2 + real1[i] ** 2))
            phases.append(atan2(imaginary1[i], real1[i]))

        for i in range(int(len(csi_raw_data2) / 2)):
            amplitudes.append(sqrt(imaginary2[i] ** 2 + real2[i] ** 2))
            phases.append(atan2(imaginary2[i], real2[i]))

        # print([csi_data[1]] + amplitudes+phases)
        # csi_list.append([csi_data[1]] + amplitudes+phases)

        # Get current date and time

        now = datetime.datetime.now()

        if count == 0:
            startTime = now

        if startTime + datetime.timedelta(seconds=1) >= datetime.datetime.now():
            count += 1
            timeString = now.strftime("%H:%M:%S.%f")
            csi_list = np.vstack(
                [csi_list, [csi_data1[1]] + [timeString] + amplitudes+phases + [csi_data1[3]] + [csi_data2[3]]])
            if count == 1:
                csi_list = np.delete(csi_list, np.s_[:1], axis=0)

        else:

            if count >= 25:
                return csi_list
            else:
                count -= 1
                csi_list = np.delete(csi_list, np.s_[:1], axis=0)
                try:
                    startTime = datetime.datetime.strptime(
                        csi_list[0, 1], "%H:%M:%S.%f")
                    startTime = datetime.datetime.combine(
                        datetime.date.today(), startTime.time())
                except:
                    print("Error in time conversion")
                    count = 0

                    csi_list = np.zeros((1, 260))

                    startTime = datetime.datetime.now()

                    continue


def getWavelet(data):
    X = data.drop(columns=['id', 'time'])
    pca_X = pd.DataFrame(pca.transform(X))
    coefficients, frequencies = pywt.cwt(
        pca_X.iloc[:, 0], scales, wavelet_name)
    wavelet_variances = np.sum(coefficients**2, axis=0)
    return wavelet_variances


def getPredictions(data):
    predicted_probs = []
    for i in range(6):
        predicted_probs.append(gmm_models[i].score_samples(data))
    predicted_labels = np.argmax(predicted_probs)
    return predicted_labels


def get25(group):
    fraction = len(group) / 25
    forPredict = []
    for i in range(1, 26):
        index = int(i * fraction)
        forPredict.append(group.iloc[index-1])

    return forPredict


def addLastQueue(data):
    global lastQueue
    if len(lastQueue) == 7:
        lastQueue.pop(0)
    lastQueue.append(data)


ifStandUpDetected = False


def lastOutput(inpt):
    global lastQueue
    global ifStandUpDetected

    if (inpt == 1):
        if not ifStandUpDetected:
            ifStandUpDetected = True
            addLastQueue(inpt)
        else:
            addLastQueue(0)

        print("Input == 1")
        return 0

    else:
        addLastQueue(inpt)
        if ifStandUpDetected:
            print("Is standup detcted :", ifStandUpDetected)
            ifStandUpDetected = False
            if inpt == 2 or inpt == 1:
                return 1
            else:
                return 0
        else:
            print("Last Queue :", lastQueue)
            most_frequent = max(set(lastQueue), key=lastQueue.count)
            return most_frequent


def getLastOutput(inpt):
    predicted = predict(inpt)
    return lastOutput(predicted)


def addRecent(data):
    global recent
    if len(recent) == 5:
        recent.pop(0)
    recent.append(data)


def predictionMapping(prediction):

    global mode

    if mode == NIGHT_TIME:

        if prediction == 0 or prediction == 4:
            return 0
        elif prediction == 5:
            return 1
        else:
            return 2

    else:

        if prediction == 0 or prediction == 4 or prediction == 5:
            return 0
        else:
            return 2


def predict(prediction):
    global recent
    prediction = predictionMapping(prediction)
    if len(recent) > 2:

        if prediction == 1:
            addRecent(prediction)
            return prediction

        else:

            if recent[len(recent)-1] == recent[len(recent)-2] == prediction:
                addRecent(prediction)
                return prediction
            elif recent[len(recent)-2] == recent[len(recent)-3] == prediction:
                addRecent(prediction)
                return prediction
            else:
                if len(recent) > 3:
                    if recent == prediction:
                        addRecent(prediction)
                        return prediction
                    else:
                        addRecent(prediction)
                        return recent[len(recent)-2]
                else:
                    addRecent(prediction)
                    return recent[len(recent)-2]
    else:
        addRecent(prediction)
        return prediction


def getPrediction(group):
    forPredict = get25(group)
    wavelet = getWavelet(pd.DataFrame(forPredict))
    wavelet = pd.DataFrame(wavelet.reshape(1, -1))
    wavelet.columns = [str(i) for i in range(0, 25)]
    pred = int(getPredictions(wavelet))
    return getLastOutput(pred)


def calibrate():

    print("Calibration Started...")

    wavelet_list = np.zeros((1, 25))

    time.sleep(2)

    for i in range(100):

        csi_data = pd.DataFrame(csi_data_read_parse())

        csi_data.columns = AMP_AND_PHASE_COLUMNS_NAMES

        forTrain = get25(csi_data)
        wavelet = getWavelet(pd.DataFrame(forTrain))
        wavelet_list = np.vstack([wavelet_list, wavelet])

        print("\tData Point Collected : ", i+1)

    wavelet_list = np.delete(wavelet_list, np.s_[:1], axis=0)

    wavelet_list = pd.DataFrame(wavelet_list)

    wavelet_list.columns = [str(i) for i in range(0, 25)]

    print("Calibrating...")

    gmm_models[0].fit(wavelet_list)

    joblib.dump(gmm_models[0], save_dir + 'gmm_model_0.pkl')

    print("Calibration Completed...")


def stopPredictions():
    global ifRun
    ifRun = False


def getPredictionsBackground():

    global ifRun
    ifRun = True
    thread = threading.Thread(target=getPredictionsBackground)
    thread.start()


def playAlarm():
    # print("Playing Alarm")
    soundPath = "alarms/classic-alarm.wav"
    os.system("aplay " + soundPath + " > /dev/null 2>&1")


def getPredictionsBackground():
    global recent
    global lastQueue
    while ifRun:
        csi_data = pd.DataFrame(csi_data_read_parse())
        # print(csi_data)

        csi_data.columns = AMP_AND_PHASE_COLUMNS_NAMES
        # print(csi_data.columns)

        pred = getPrediction(csi_data)

        print("\tTime: " + str(datetime.datetime.now()), end="")

        print("\tPrediction:  ACTIVITY" + str(pred))

        if (pred == 1):
            recent = [0, 0, 0, 0, 0]
            lastQueue = [0, 0, 0, 0, 0]
            send_notification("Wakeup Detected",
                              "System will sleep for 5 minutes")
            time.sleep(60*5)

        if (pred == 2):
            print("\t\tIntruder Detected.")
            send_notification("Intruder Detected", "Intruder Detected")
            playAlarm()
            # print("Intruder Detected")
            recent = [0, 0, 0, 0, 0]
            lastQueue = [0, 0, 0, 0, 0]
            time.sleep(30)


def startPredictions():
    global ifRun
    ifRun = True
    thread = threading.Thread(target=getPredictionsBackground)
    thread.start()


def changeModeToNight():
    global mode
    mode = NIGHT_TIME


def changeModeToDay():
    global mode
    mode = DAY_TIME
