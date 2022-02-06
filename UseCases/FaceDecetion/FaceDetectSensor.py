import socket
import time
from time import process_time
import cv2
import numpy as np
import glob
import sys
import os
import csv
import random
import tracemalloc
from skimage.util import random_noise

############################################## Knob settings ############################################
qos_format = str(sys.argv[1])  # select from the given choices
qos_val = int(sys.argv[2])  # check the paper for range
filtering = int(sys.argv[3])  # 1 or 0
noise_name = str(sys.argv[4])  # select from the given choices
noise_intesity = float(sys.argv[5])  # degree of your choice

config_param = qos_format+'_'+str(qos_val)+'_'+str(filtering)
print(config_param)
############################################## Knob settings ############################################
HEADERSIZE = 10
host = ''  # hostname
print(host)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

while(True):
    state = s.getsockopt( socket.SOL_SOCKET, socket.SO_REUSEADDR )
    print("State: ", state)
    if(state==1):
        print("Port is free now")
        break

port=12345
s.bind(('', port))
s.listen(5)
clientsocket, address = s.accept()

print(f"Connection from {address} has been established.")

##################### Time count start ########################
# Start the stopwatch / counter
tracemalloc.start()
t1_start = process_time()
start = time.time()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
dataSizeTotal = 0
count = 0
tobe_sent = []


def noise_maker(image, noise_type, intesity):
    noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    if noise_type == "uni":
        cv2.randu(noise, 0, 255)
        uniform_noise = (noise*intesity).astype(np.uint8)
        noisy_image = cv2.add(image, uniform_noise)

    elif noise_type == "imp":
        cv2.randu(noise, 0, 255)
        uniform_noise = (noise*intesity).astype(np.uint8)
        _, impulse_noise = cv2.threshold(
            uniform_noise, 120, 255, cv2.THRESH_BINARY)
        noisy_image = cv2.add(image, impulse_noise)
        # noisy_image = sp_noise(image, intesity-0.1)
    elif noise_type == "gaus":
        noise = 30 * np.random.randn(*image.shape)  # effective
        noisy_image = np.uint8(np.clip(image + noise, 0, 255))
    elif noise_type == "localvar":
        localvar_noise = random_noise(image, mode='localvar')
        noisy_image = np.array(255*localvar_noise, dtype='uint8')
    elif noise_type == "speckle":
        gauss = np.random.normal(0, 1, image.size)
        gauss = gauss.reshape(
            image.shape[0], image.shape[1]).astype('uint8')
        noisy_image = image + image * gauss
    return noisy_image


def image_compression(image):
    # The function imencode compresses the image.
    if qos_format == '.png':
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), qos_val]
        result, img_encoded = cv2.imencode(qos_format, image, encode_param)

    elif qos_format == '.jpg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qos_val]
        result, img_encoded = cv2.imencode(qos_format, image, encode_param)

    elif qos_format == '.webp':
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), qos_val]
        result, img_encoded = cv2.imencode(qos_format, image, encode_param)

    if result == False:
        print("could not encode image!")
        quit()

    data = img_encoded.tostring()  # <class 'bytes'>
    return data


def transfer_data(data):
    global dataSizeTotal
    global count
    dataSize = len(data) / 1024
    dataSizeTotal = dataSizeTotal + dataSize

    #print("data type before:", type(data))

    clientsocket.send(bytes(f"{len(data):<{HEADERSIZE}}", 'utf-8')+data)

    print("Sent frame: ", count)
    count = count + 1
    time.sleep(1)


def pre_processor(images):
    for frame in images:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if filtering:
            faces = face_cascade.detectMultiScale(
                frame, scaleFactor=1.01,  minSize=(10, 10))
            if len(faces) != 0:
                noisy_img = noise_maker(
                    gray_img, str(noise_name), noise_intesity)
                data = image_compression(noisy_img)
                tobe_sent.append(data)
                # transfer_data(data)
                time.sleep(.2)
            else:
                img = np.zeros((10, 10, 1), np.uint8)
                data = image_compression(img)
                tobe_sent.append(data)
        else:
            noisy_img = noise_maker(gray_img, str(noise_name), noise_intesity)
            data = image_compression(noisy_img)
            tobe_sent.append(data)
            # transfer_data(data)


images = [cv2.imread(file) for file in sorted(
    glob.glob("Your image dataset"))]
print("Read the images")

pre_processor(images)
print("Done pre-processing")

first_stat = str(len(tobe_sent))+','+config_param

clientsocket.send(
    bytes(f"{len(first_stat):<{HEADERSIZE}}"+first_stat, "utf-8"))

for frames in tobe_sent:
    transfer_data(frames)
print("Sent Frames")

############ Stop the stopwatch / counter ################
t1_stop = process_time()
passed = t1_stop-t1_start
elapsed = time.time() - start
_, first_peak = tracemalloc.get_traced_memory()
memoryUSed = first_peak/10**6

stat = str(round(elapsed, 2))+','+str(round(passed, 2))+','+str(round(memoryUSed, 2))+',' +\
    str(round(dataSizeTotal, 2))+','+str(cpuUtilized)
clientsocket.sendall(
    bytes(f"{len(stat):<{HEADERSIZE}}"+stat, "utf-8"))
print("Total sent frames: ", count)

s.close()
clientsocket.close()
os._exit(0)
