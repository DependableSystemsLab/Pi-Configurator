import socket
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import wave
import json
import numpy as np
from scipy import signal
import json
#import speech_recognition as sr
from difflib import SequenceMatcher
import tracemalloc
import sys
import os
import time
from time import process_time
import csv
import deepspeech

width = int(sys.argv[1])
rate = float(sys.argv[2])
proc_param = 'W:_'+str(width)+'_R:_'+str(rate)


model_file_path = 'deepspeech-0.9.3-models.tflite'
model = deepspeech.Model(model_file_path)
scorer_file_path = 'deepspeech-0.9.3-models.scorer'
model.enableExternalScorer(scorer_file_path)

lm_alpha = rate
lm_beta = 1.85  # can be controlled as a knob like mentioned in the paper.
model.setScorerAlphaBeta(lm_alpha, lm_beta)
beam_width = width  # [1..500]
model.setBeamWidth(beam_width)

HEADERSIZE = 10
host = ''  # your audio sensor IP address.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

connected = False
while not connected:
    try:
        s.connect((host, 12345))
        connected = True
    except Exception as e:
        print(e)  # Do nothing, just try again
print("Connection established")


def csvWriter(sen_param, proc_param, ProWallTime, ProCPUTime, ProMem, textComp, SenWallTime, SenCPUTime, SenMem, TransVol, GlobalWallTime, GlobalCPUTime, GlobalMem):
    with open("AudioStat.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        ret = writer.writerow(
            [sen_param, proc_param, ProWallTime, ProCPUTime, ProMem, textComp, SenWallTime, SenCPUTime, SenMem, TransVol, GlobalWallTime, GlobalCPUTime, GlobalMem])
        return ret


def textRecord(AUDIO_FILE):
    #sampleRate, buffer = read(AUDIO_FILE)
    w = wave.open(AUDIO_FILE, 'rb')
    sampleRate = w.getframerate()
    frames = w.getnframes()
    buff = w.readframes(frames)
    data16 = np.frombuffer(buff, dtype=np.int16)
    text = model.stt(data16)
    return text


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def similarityScore(audio_data):
    print("Starting the similarity checking")
    newtext = textRecord('ProdAudio.wav')
    textSimilarity = round(similar(orgText, newtext), 2)

    if beam_width <= 100:
        n = beam_width
    elif ((beam_width > 100) and (beam_width <= 500)):
        n = round(beam_width/25)
    elif ((beam_width > 500) and (beam_width <= 1000)):
        n = round(beam_width/10)

    return textSimilarity


def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def processor(audio_numpy, stat):
    print("Starting to process")
    incoming_stat = stat.split(',')
    sen_param = str(incoming_stat[0])
    sensor_wallTime = float(incoming_stat[1])
    sensor_cpuTime = float(incoming_stat[2])
    sensor_mem = float(incoming_stat[3])
    sensor_size = float(incoming_stat[4])

    y = float2pcm(audio_numpy)

    # writing the sensor received audio file
    write("ProdAudio.wav", comp_sample_rate, y)
    print("Written the received audio file")

    #textSimilarity, snr = similarityScore(y)
    textSimilarity = similarityScore(y)

    ########## Stop the stopwatch / counter#########
    t1_stop = process_time()
    cpuTime = round((t1_stop-t1_start), 2)
    end = time.time()
    wallTime = round((end - start), 2)
    _, first_peak = tracemalloc.get_traced_memory()
    memoryUSed = round((first_peak/10**6), 2)

    ret = csvWriter(sen_param, proc_param, wallTime, cpuTime, memoryUSed, textSimilarity, sensor_wallTime,
                    sensor_cpuTime, sensor_mem, sensor_size, wallTime+sensor_wallTime, cpuTime+sensor_cpuTime, memoryUSed+sensor_mem)
    return ret


print("Going to receive the audio now ISA")
terminate_counter = 0
BUFFER_SIZE = 50000
while True:
    received_msg = b''
    rceived_msg_status = True
    iteration = 1
    while True:
        terminate_counter = terminate_counter+1
        if terminate_counter > 555000:
            print("Frames will not arrived")
            s.close()
            os.remove("ProdAudio.wav")
            os._exit(0)

        first_new_msg = s.recv(BUFFER_SIZE)

        if rceived_msg_status and len(first_new_msg[:HEADERSIZE]) > 0:
            msglen = int(first_new_msg[:HEADERSIZE])
            rceived_msg_status = False

        received_msg += first_new_msg

        if iteration == 1 and len(received_msg)-HEADERSIZE == msglen:
            ############## Time Count starts ###################
            # Start the stopwatch / counter
            tracemalloc.start()
            t1_start = process_time()
            start = time.time()

            comp_sample_rate = int(
                received_msg[HEADERSIZE:])
            print("Got the sample rate")

            received_msg = b''
            rceived_msg_status = True
            iteration = iteration + 1

        # full message recvd
        elif iteration == 2 and len(received_msg)-HEADERSIZE >= msglen:
            received_audio_data = received_msg[HEADERSIZE:msglen+HEADERSIZE]
            print("Got the audio")
            stat = received_msg[msglen+2*HEADERSIZE:].decode('utf8')
            print("Got the stats")
            break
    break
s.close()

file = open("audioSpeech.txt", "r")
orgText = file.read()
file.close()

audio_numpy = np.array(json.loads(received_audio_data))
ret = processor(audio_numpy, stat)
if ret > 0:
    print("Done processing")
    os.remove("ProdAudio.wav")
    os._exit(0)
