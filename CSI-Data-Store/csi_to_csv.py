import time
import serial
import csv
import json
import numpy as np
from io import StringIO
from math import sqrt, atan2
from datetime import datetime

CSI_VAID_SUBCARRIER_INTERVAL = 3

# Remove invalid subcarriers
# secondary channel : below, HT, 40 MHz, non STBC, v, HT-LFT: 0~63, -64~-1, 384
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_color = []
color_step = 255 // (28 // CSI_VAID_SUBCARRIER_INTERVAL + 1)

# LLTF: 52
# 26  red
csi_vaid_subcarrier_index += [i for i in range(
    6, 32, CSI_VAID_SUBCARRIER_INTERVAL)]
csi_vaid_subcarrier_color += [(i * color_step, 0, 0)
                              for i in range(1,  26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
# 26  green
csi_vaid_subcarrier_index += [i for i in range(
    33, 59, CSI_VAID_SUBCARRIER_INTERVAL)]
csi_vaid_subcarrier_color += [(0, i * color_step, 0)
                              for i in range(1,  26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
CSI_DATA_LLFT_COLUMNS = len(csi_vaid_subcarrier_index)

# HT-LFT: 56 + 56
# 28  blue
csi_vaid_subcarrier_index += [i for i in range(
    66, 94, CSI_VAID_SUBCARRIER_INTERVAL)]
csi_vaid_subcarrier_color += [(0, 0, i * color_step)
                              for i in range(1,  28 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
# 28  White
csi_vaid_subcarrier_index += [i for i in range(
    95, 123, CSI_VAID_SUBCARRIER_INTERVAL)]
csi_vaid_subcarrier_color += [(i * color_step, i * color_step, i * color_step)
                              for i in range(1,  28 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]

CSI_DATA_INDEX = 200  # buffer size
CSI_DATA_COLUMNS = len(csi_vaid_subcarrier_index)
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]

AMP_AND_PHASE_COLUMNS_NAMES = ['id', 'time', 'amplitude1_1', 'amplitude2_1', 'amplitude3_1', 'amplitude4_1', 'amplitude5_1', 'amplitude6_1', 'amplitude7_1', 'amplitude8_1', 'amplitude9_1', 'amplitude10_1', 'amplitude11_1', 'amplitude12_1', 'amplitude13_1', 'amplitude14_1', 'amplitude15_1', 'amplitude16_1', 'amplitude17_1', 'amplitude18_1', 'amplitude19_1', 'amplitude20_1', 'amplitude21_1', 'amplitude22_1', 'amplitude23_1', 'amplitude24_1', 'amplitude25_1', 'amplitude26_1', 'amplitude27_1', 'amplitude28_1', 'amplitude29_1', 'amplitude30_1', 'amplitude31_1', 'amplitude32_1', 'amplitude33_1', 'amplitude34_1', 'amplitude35_1', 'amplitude36_1', 'amplitude37_1', 'amplitude38_1', 'amplitude39_1', 'amplitude40_1', 'amplitude41_1', 'amplitude42_1', 'amplitude43_1', 'amplitude44_1', 'amplitude45_1', 'amplitude46_1', 'amplitude47_1', 'amplitude48_1', 'amplitude49_1', 'amplitude50_1', 'amplitude51_1', 'amplitude52_1', 'amplitude53_1', 'amplitude54_1', 'amplitude55_1', 'amplitude56_1', 'amplitude57_1', 'amplitude58_1', 'amplitude59_1', 'amplitude60_1_1', 'amplitude61_1', 'amplitude62_1', 'amplitude63_1', 'amplitude64_1', 'phase1_1', 'phase2_1', 'phase3_1', 'phase4_1', 'phase5_1', 'phase6_1', 'phase7_1', 'phase8_1', 'phase9_1', 'phase10_1', 'phase11_1', 'phase12_1', 'phase13_1', 'phase14_1', 'phase15_1', 'phase16_1', 'phase17_1', 'phase18_1', 'phase19_1', 'phase20_1', 'phase21_1', 'phase22_1', 'phase23_1', 'phase24_1', 'phase25_1', 'phase26_1', 'phase27_1', 'phase28_1', 'phase29_1', 'phase30_1', 'phase31_1', 'phase32_1', 'phase33_1', 'phase34_1', 'phase35_1', 'phase36_1', 'phase37_1', 'phase38_1', 'phase39_1', 'phase40_1', 'phase41_1', 'phase42_1', 'phase43_1', 'phase44_1', 'phase45_1', 'phase46_1', 'phase47_1', 'phase48_1', 'phase49_1', 'phase50_1', 'phase51_1', 'phase52_1', 'phase53_1', 'phase54_1', 'phase55_1', 'phase56_1', 'phase57_1', 'phase58_1', 'phase59_1', 'phase60_1', 'phase61_1', 'phase62_1', 'phase63_1',
                               'phase64_1', 'amplitude1_2', 'amplitude2_2', 'amplitude3_2', 'amplitude4_2', 'amplitude5_2', 'amplitude6_2', 'amplitude7_2', 'amplitude8_2', 'amplitude9_2', 'amplitude10_2', 'amplitude11_2', 'amplitude12_2', 'amplitude13_2', 'amplitude14_2', 'amplitude15_2', 'amplitude16_2', 'amplitude17_2', 'amplitude18_2', 'amplitude19_2', 'amplitude20_2', 'amplitude21_2', 'amplitude22_2', 'amplitude23_2', 'amplitude24_2', 'amplitude25_2', 'amplitude26_2', 'amplitude27_2', 'amplitude28_2', 'amplitude29_2', 'amplitude30_2', 'amplitude31_2', 'amplitude32_2', 'amplitude33_2', 'amplitude34_2', 'amplitude35_2', 'amplitude36_2', 'amplitude37_2', 'amplitude38_2', 'amplitude39_2', 'amplitude40_2', 'amplitude41_2', 'amplitude42_2', 'amplitude43_2', 'amplitude44_2', 'amplitude45_2', 'amplitude46_2', 'amplitude47_2', 'amplitude48_2', 'amplitude49_2', 'amplitude50_2', 'amplitude51_2', 'amplitude52_2', 'amplitude53_2', 'amplitude54_2', 'amplitude55_2', 'amplitude56_2', 'amplitude57_2', 'amplitude58_2', 'amplitude59_2', 'amplitude60_2', 'amplitude61_2', 'amplitude62_2', 'amplitude63_2', 'amplitude64_2', 'phase1_2', 'phase2_2', 'phase3_2', 'phase4_2', 'phase5_2', 'phase6_2', 'phase7_2', 'phase8_2', 'phase9_2', 'phase10_2', 'phase11_2', 'phase12_2', 'phase13_2', 'phase14_2', 'phase15_2', 'phase16_2', 'phase17_2', 'phase18_2', 'phase19_2', 'phase20_2', 'phase21_2', 'phase22_2', 'phase23_2', 'phase24_2', 'phase25_2', 'phase26_2', 'phase27_2', 'phase28_2', 'phase29_2', 'phase30_2', 'phase31_2', 'phase32_2', 'phase33_2', 'phase34_2', 'phase35_2', 'phase36_2', 'phase37_2', 'phase38_2', 'phase39_2', 'phase40_2', 'phase41_2', 'phase42_2', 'phase43_2', 'phase44_2', 'phase45_2', 'phase46_2', 'phase47_2', 'phase48_2', 'phase49_2', 'phase50_2', 'phase51_2', 'phase52_2', 'phase53_2', 'phase54_2', 'phase55_2', 'phase56_2', 'phase57_2', 'phase58_2', 'phase59_2', 'phase60_2', 'phase61_2', 'phase62_2', 'phase63_2', 'phase64_2', 'rssi_1', 'rssi_2' 'activity']
csi_data_array = np.zeros(
    [CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)


SERIAL_PORT_1 = "COM3"
SERIAL_PORT_2 = "COM8"

AMP_AND_PHASE_COLUMNS_NAMES = ['id', 'time', 'amplitude1_1', 'amplitude2_1', 'amplitude3_1', 'amplitude4_1', 'amplitude5_1', 'amplitude6_1', 'amplitude7_1', 'amplitude8_1', 'amplitude9_1', 'amplitude10_1', 'amplitude11_1', 'amplitude12_1', 'amplitude13_1', 'amplitude14_1', 'amplitude15_1', 'amplitude16_1', 'amplitude17_1', 'amplitude18_1', 'amplitude19_1', 'amplitude20_1', 'amplitude21_1', 'amplitude22_1', 'amplitude23_1', 'amplitude24_1', 'amplitude25_1', 'amplitude26_1', 'amplitude27_1', 'amplitude28_1', 'amplitude29_1', 'amplitude30_1', 'amplitude31_1', 'amplitude32_1', 'amplitude33_1', 'amplitude34_1', 'amplitude35_1', 'amplitude36_1', 'amplitude37_1', 'amplitude38_1', 'amplitude39_1', 'amplitude40_1', 'amplitude41_1', 'amplitude42_1', 'amplitude43_1', 'amplitude44_1', 'amplitude45_1', 'amplitude46_1', 'amplitude47_1', 'amplitude48_1', 'amplitude49_1', 'amplitude50_1', 'amplitude51_1', 'amplitude52_1', 'amplitude53_1', 'amplitude54_1', 'amplitude55_1', 'amplitude56_1', 'amplitude57_1', 'amplitude58_1', 'amplitude59_1', 'amplitude60_1_1', 'amplitude61_1', 'amplitude62_1', 'amplitude63_1', 'amplitude64_1', 'phase1_1', 'phase2_1', 'phase3_1', 'phase4_1', 'phase5_1', 'phase6_1', 'phase7_1', 'phase8_1', 'phase9_1', 'phase10_1', 'phase11_1', 'phase12_1', 'phase13_1', 'phase14_1', 'phase15_1', 'phase16_1', 'phase17_1', 'phase18_1', 'phase19_1', 'phase20_1', 'phase21_1', 'phase22_1', 'phase23_1', 'phase24_1', 'phase25_1', 'phase26_1', 'phase27_1', 'phase28_1', 'phase29_1', 'phase30_1', 'phase31_1', 'phase32_1', 'phase33_1', 'phase34_1', 'phase35_1', 'phase36_1', 'phase37_1', 'phase38_1', 'phase39_1', 'phase40_1', 'phase41_1', 'phase42_1', 'phase43_1', 'phase44_1', 'phase45_1', 'phase46_1', 'phase47_1', 'phase48_1', 'phase49_1', 'phase50_1', 'phase51_1', 'phase52_1', 'phase53_1', 'phase54_1', 'phase55_1', 'phase56_1', 'phase57_1', 'phase58_1', 'phase59_1', 'phase60_1', 'phase61_1', 'phase62_1', 'phase63_1',
                               'phase64_1', 'amplitude1_2', 'amplitude2_2', 'amplitude3_2', 'amplitude4_2', 'amplitude5_2', 'amplitude6_2', 'amplitude7_2', 'amplitude8_2', 'amplitude9_2', 'amplitude10_2', 'amplitude11_2', 'amplitude12_2', 'amplitude13_2', 'amplitude14_2', 'amplitude15_2', 'amplitude16_2', 'amplitude17_2', 'amplitude18_2', 'amplitude19_2', 'amplitude20_2', 'amplitude21_2', 'amplitude22_2', 'amplitude23_2', 'amplitude24_2', 'amplitude25_2', 'amplitude26_2', 'amplitude27_2', 'amplitude28_2', 'amplitude29_2', 'amplitude30_2', 'amplitude31_2', 'amplitude32_2', 'amplitude33_2', 'amplitude34_2', 'amplitude35_2', 'amplitude36_2', 'amplitude37_2', 'amplitude38_2', 'amplitude39_2', 'amplitude40_2', 'amplitude41_2', 'amplitude42_2', 'amplitude43_2', 'amplitude44_2', 'amplitude45_2', 'amplitude46_2', 'amplitude47_2', 'amplitude48_2', 'amplitude49_2', 'amplitude50_2', 'amplitude51_2', 'amplitude52_2', 'amplitude53_2', 'amplitude54_2', 'amplitude55_2', 'amplitude56_2', 'amplitude57_2', 'amplitude58_2', 'amplitude59_2', 'amplitude60_2', 'amplitude61_2', 'amplitude62_2', 'amplitude63_2', 'amplitude64_2', 'phase1_2', 'phase2_2', 'phase3_2', 'phase4_2', 'phase5_2', 'phase6_2', 'phase7_2', 'phase8_2', 'phase9_2', 'phase10_2', 'phase11_2', 'phase12_2', 'phase13_2', 'phase14_2', 'phase15_2', 'phase16_2', 'phase17_2', 'phase18_2', 'phase19_2', 'phase20_2', 'phase21_2', 'phase22_2', 'phase23_2', 'phase24_2', 'phase25_2', 'phase26_2', 'phase27_2', 'phase28_2', 'phase29_2', 'phase30_2', 'phase31_2', 'phase32_2', 'phase33_2', 'phase34_2', 'phase35_2', 'phase36_2', 'phase37_2', 'phase38_2', 'phase39_2', 'phase40_2', 'phase41_2', 'phase42_2', 'phase43_2', 'phase44_2', 'phase45_2', 'phase46_2', 'phase47_2', 'phase48_2', 'phase49_2', 'phase50_2', 'phase51_2', 'phase52_2', 'phase53_2', 'phase54_2', 'phase55_2', 'phase56_2', 'phase57_2', 'phase58_2', 'phase59_2', 'phase60_2', 'phase61_2', 'phase62_2', 'phase63_2', 'phase64_2', 'rssi_1', 'rssi_2', 'activity']


# Change desired file name
FILE_NAME = "csi_data.csv"


file = open(FILE_NAME, "w")

writer = csv.writer(file, lineterminator='\n')

writer.writerow(AMP_AND_PHASE_COLUMNS_NAMES)

file.close()

time.sleep(10)


def csi_data_read_parse(ser1, ser2, activty=0):

    if ser1.isOpen():
        print("open success SER 1")
    else:
        print("open failed SER 1")

    if ser2.isOpen():
        print("open success SER 2")
    else:
        print("open failed SER 2")

    count = 0

    csi_list = np.zeros((1, 261))

    while count < 200:
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
            print("element number is not equal")
            continue

        try:
            csi_raw_data1 = json.loads(csi_data1[-1])
            csi_raw_data2 = json.loads(csi_data2[-1])
        except json.JSONDecodeError:
            print(f"data is incomplete")
            continue

        if len(csi_raw_data1) != 128 and len(csi_raw_data1) != 256 and len(csi_raw_data1) != 384:
            print(f"element number is not equal: {len(csi_raw_data1)}")
            continue

        if len(csi_raw_data2) != 128 and len(csi_raw_data2) != 256 and len(csi_raw_data2) != 384:
            print(f"element number is not equal: {len(csi_raw_data2)}")
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
        now = datetime.now()

        # Format time as a string
        time_string = now.strftime("%H:%M:%S")

        csi_list = np.vstack(
            [csi_list, [csi_data1[1]] + [time_string] + amplitudes+phases + [csi_data1[3]] + [csi_data2[3]] + [activty]])
        count += 1
        # print(count)

    csi_list = np.delete(csi_list, np.s_[:1], axis=0)

    return csi_list


STATIC = 0
WALKING = 1
BENDING = 2
CRAWLING = 3
SLEEPING = 4
WAKEUP = 5


a = 0

while True:

    # try:
    ser1 = serial.Serial(port=SERIAL_PORT_1, baudrate=921600,
                         bytesize=8, parity='N', stopbits=1)

    ser2 = serial.Serial(port=SERIAL_PORT_2, baudrate=921600,
                         bytesize=8, parity='N', stopbits=1)
    f = open(FILE_NAME, "a")
    w = csv.writer(f, lineterminator='\n')

    csi_data = csi_data_read_parse(ser1, ser2, WAKEUP)

    a += 1

    w.writerows(csi_data)
    f.close()

    print("Data Written")
    ser2.close()
    ser1.close()

    if a == 100:
        print("Wakeup Done")
        break
