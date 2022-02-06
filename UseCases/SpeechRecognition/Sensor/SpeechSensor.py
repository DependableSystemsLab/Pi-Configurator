import socket
from scipy import signal
from scipy.io.wavfile import read
import json
import numpy as np
from scipy.fftpack import fft
import tracemalloc
import os
import time
from time import process_time
import librosa
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

qos_val = 5  # adjust compression rate

need_Reconstruct = int(sys.argv[1])
config_param = 'Q5_Filt?:_' + str(need_Reconstruct)+'_'
if need_Reconstruct == True:
    filter_param = str(sys.argv[2])
    filter_arg = int(sys.argv[3])

    config_param = config_param+'F:_' + filter_param+'P:_'+str(filter_arg)


HEADERSIZE = 10
host = socket.gethostname()
print(host)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

while(True):
    state = s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
    print("State: ", state)
    if(state == 1):
        print("Port is free now")
        break

s.bind((host, 12345))
s.listen(5)
clientsocket, address = s.accept()

print(f"Connection from {address} has been established.")

##################### Time count start ########################
# Start the stopwatch / counter
tracemalloc.start()
t1_start = process_time()
start = time.time()


def compress(audioArray, targetSR, orgSR):
    print("Starting to compress")
    data = signal.resample_poly(audioArray, targetSR, orgSR)
    # Convert `data` to 32 bit integers:
    y = (np.iinfo(np.int32).max * (audioArray /
                                   np.abs(audioArray).max())).astype(np.int32)
    return data


def addNoise(array):
    # Calculating the fourier transformation of the signal
    FourierTransformation = fft(array)
    # Adding guassian Noise to the signal.
    GuassianNoise = 10*np.random.rand(len(FourierTransformation))
    NoisySound = GuassianNoise + array
    print("Noise is loaded")
    return NoisySound


def ButterWorthFilter(NoisyData, FilterOrder, sampleRate):
    # ButterWorth high-filter
    b, a = signal.butter(FilterOrder, 1000/(sampleRate/2), btype='highpass')
    filteredSignal = signal.lfilter(b, a, NoisyData)
    # ButterWorth low-filter
    c, d = signal.butter(FilterOrder, 380/(sampleRate/2), btype='lowpass')
    # Applying the filter to the signal
    FilteredSignal = signal.lfilter(c, d, filteredSignal)

    return FilteredSignal


def running_mean(x, windowSize):
    cumsum = np.cumsum(np.insert(x, 0, 0))

    if windowSize <= 100:
        n = windowSize
    elif ((windowSize > 100) and (windowSize <= 500)):
        n = round(windowSize/2)
    elif ((windowSize > 500) and (windowSize <= 1000)):
        n = round(windowSize/10)
    elif ((windowSize > 1000) and (windowSize <= 1500)):
        n = round(windowSize/200)
    elif windowSize > 1500:
        n = round(windowSize/50)

    return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize


def audioFilter(audio_numpy, sr):
    print("Going to filter now")
    if filter_param == 'butter':
        newFilteredSignal = ButterWorthFilter(
            audio_numpy, filter_arg, sr)  # range [1..15]

    elif filter_param == 'running':
        newFilteredSignal = running_mean(
            audio_numpy, filter_arg)  # range [1..2000]

    return newFilteredSignal


print("Going to read audio file")
audioFile = ''  # your audio data source
sampleRate, data = read(audioFile)
print("Original sample rate", sampleRate)

audio = addNoise(data)

target_sr = round(sampleRate/qos_val)  # compressed SF
audio = compress(audio, target_sr, sampleRate)
sampleRate = target_sr

if need_Reconstruct == True:
    audio = librosa.effects.preemphasis(audio)
    audio = audioFilter(audio, sampleRate)

sampleRate = str(sampleRate)
signal_str = json.dumps(audio.tolist())

audio_info = bytes(
    f"{len(sampleRate):<{HEADERSIZE}}"+sampleRate, "utf-8")
audio_data = bytes(f"{len(signal_str):<{HEADERSIZE}}"+signal_str, 'utf-8')

print("Sending the data")
clientsocket.send(audio_info)
clientsocket.send(audio_data)
print("Audio sent")

############ Stop the stopwatch / counter ################
t1_stop = process_time()
_, first_peak = tracemalloc.get_traced_memory()
wallTime = time.time() - start
cpuTime = t1_stop-t1_start
memoryUsed = first_peak/10**6

stat = config_param+','+str(round(wallTime, 2))+','+str(round(cpuTime, 2))+',' + \
    str(round(memoryUsed, 2))+',' + str(audio.size)
clientsocket.send(
    bytes(f"{len(stat):<{HEADERSIZE}}"+stat, "utf-8"))

print("All the data are sent")
s.close()
clientsocket.close()
# os._exit(0)
