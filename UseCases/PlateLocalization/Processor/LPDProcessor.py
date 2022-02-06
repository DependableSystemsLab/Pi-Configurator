from cv2 import cv2
import statistics
import glob
import csv
import numpy as np
import time
from time import process_time
from skimage.metrics import structural_similarity as ssim
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio as psnr
import socket
import sys
import os
import tracemalloc
import imutils

# This is the processor
print("Starting")
############################################## Knob settings ############################################
filterSignal = int(sys.argv[1])  # intial pre-filtering 1/0

need_Reconstruct = int(sys.argv[2])
# Contrast limited adaptive histogram equalization.
hist_eq = int(sys.argv[3])

config_param = str(filterSignal)+'_' + \
    str(need_Reconstruct)+'_'+str(hist_eq)+'_'

if need_Reconstruct == True:
    filter_param = str(sys.argv[4])
    filter_arg = int(sys.argv[5])
    # tileSize = int(sys.argv[6]) #can be enabled for contrast correction

    config_param = config_param+'_'+filter_param+'_'+str(filter_arg)
print(config_param)
############################################## Knob settings ############################################
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tileSize, tileSize))
# clipLimit can be used to adjust contrast.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1, 1))
str1 = './PlateTruth/'
str2 = './PlateTest/'
############################################## File Cleaning ############################################
files = glob.glob('./PlateTest/*.png')
if len(files) > 0:
    for f in files:
        os.remove(f)
############################################## File Cleaning ############################################

HEADERSIZE = 10
host = ' '  # your sensor IP address
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

connected = False
while not connected:
    try:
        s.connect((host, 12345))
        connected = True
    except Exception as e:
        print(e)  # Do nothing, just try again
print("Connection established")


def csvWriter(sen_param, proc_param, ProWallTime, ProCPUTime, ProMem, ProSSIM, SenWallTime, SenCPUTime, SenMem, TransVol, GlobalWallTime, GlobalCPUTime, GlobalMem):
    with open("Stat.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        ret = writer.writerow(
            [sen_param, proc_param, ProWallTime, ProCPUTime, ProMem, ProSSIM, SenWallTime, SenCPUTime, SenMem, TransVol, GlobalWallTime, GlobalCPUTime, GlobalMem])
        return ret


def compare_face(str1, str2):
    count = 0
    scoreSetImageMean_ssim = []
    mean_ssim = 0

    while count < limit:
        count = count+1
        imageSet1 = []
        imageSet2 = []
        pattern = 'Plate_'+str(count)+'.png'
        imageSet1 = [cv2.imread(file) for file in glob.glob(str1+pattern)]
        imageSet2 = [cv2.imread(file) for file in glob.glob(str2+pattern)]

        scoreSetFrame_ssim = []
        for frame1 in imageSet1:
            grayA = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            max_score_ssim = 0
            for frame2 in imageSet2:
                grayB = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                if grayA.shape[1] != grayB.shape[1] and grayA.shape[0] != grayB.shape[0]:
                    dim = (grayA.shape[1], grayA.shape[0])
                    grayB = cv2.resize(
                        grayB, dim, interpolation=cv2.INTER_CUBIC)

                try:
                    score_ssim = ssim(grayA, grayB)
                    if score_ssim > max_score_ssim:
                        max_score_ssim = score_ssim
                except ValueError:
                    pass

            scoreSetFrame_ssim.append(max_score_ssim)

        try:
            scoreSetImageMean_ssim.append(statistics.mean(scoreSetFrame_ssim))
        except statistics.StatisticsError:
            pass

    try:
        mean_ssim = statistics.mean(scoreSetImageMean_ssim)
    except statistics.StatisticsError:
        pass

    return mean_ssim


def image_reconstructor(image):
    if filter_param == 'non':
        dst = cv2.fastNlMeansDenoising(image, 10, 21, filter_arg)

    elif filter_param == 'gaus':
        dst = cv2.GaussianBlur(image, (filter_arg, filter_arg), 0)

    if hist_eq == True:
        dst = clahe.apply(dst)
    return dst


def plate_Detector(image, frameCount):
    image = imutils.resize(image, width=500)
    if filterSignal == 1:
        image = cv2.bilateralFilter(image, 11, 90, 90)

    high_thresh, _ = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    edges = cv2.Canny(image, lowThresh, high_thresh)

    cnts, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    plate = None
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(edges_count) == 4:
            x, y, w, h = cv2.boundingRect(c)
            if w > h and h < .75*w and h > .25*w:
                plate = image[y:y+h, x:x+w]
                cv2.imwrite(
                    './PlateTest/Plate_'+str(frameCount)+'.png', plate)
                break


def process_image():
    frameCount = 0
    for img in images:
        if need_Reconstruct == True:
            if np.all(img == 8):
                frameCount = frameCount+1
                print('Got empty one......................................')
                continue
            else:
                frameCount = frameCount+1
                img_Reconstructed = image_reconstructor(img)
                plate_Detector(img_Reconstructed, frameCount)
                # print("Processed frame: ", frameCount)
        else:
            if np.all(img == 8):
                frameCount = frameCount+1
                print('Got empty one')
                continue
            else:
                frameCount = frameCount+1
                plate_Detector(img, frameCount)
                print("Processed frame: ", frameCount)

    if limit == frameCount:
        print("Now")
        ########## Stop the stopwatch / counter#########
        t1_stop = process_time()
        passed = round((t1_stop-t1_start), 2)
        end = time.time()
        elapsed = round((end - start), 2)
        _, first_peak = tracemalloc.get_traced_memory()
        memoryUSed = round((first_peak/10**6), 2)

        meanFrameScore_ssim1 = compare_face(str1, str2)
        meanFrameScore_ssim2 = compare_face(str2, str1)
        ssim_face = round(
            ((meanFrameScore_ssim1 + meanFrameScore_ssim2)/2), 2)

        ret = csvWriter(incoming_param, config_param, elapsed, passed, memoryUSed, ssim_face*100, sensor_time,
                        sensor_time2, sensor_mem, sensor_size, elapsed+sensor_time, passed+sensor_time2, memoryUSed+sensor_mem)
        if ret > 0:
            s.close()
            os._exit(0)


def sortKeyFunc(s):
    return int(os.path.basename(s)[5:-4])


images = []
receivedCount = 0
terminate_counter = 0
while True:
    received_msg = b''
    received_msg_status = True
    iteration = 1

    while True:
        terminate_counter = terminate_counter+1
        if terminate_counter > 15000:
            print("Frames will not arrived")
            # tracemalloc.reset_peak()
            s.close()
            os._exit(0)

        first_new_msg = s.recv(10000)

        if received_msg_status and len(first_new_msg[:HEADERSIZE]) > 0:
            recived_msglen = int(first_new_msg[:HEADERSIZE])
            received_msg_status = False

        received_msg += first_new_msg

        if iteration == 1 and len(received_msg)-HEADERSIZE == recived_msglen:

            ############## Time Count starts ###################
            # Start the stopwatch / counter
            tracemalloc.start()
            t1_start = process_time()
            start = time.time()

            incoming_stat = received_msg[HEADERSIZE:].decode('utf8')
            first_stat = incoming_stat.split(',')
            limit = int(first_stat[0])
            incoming_param = str(first_stat[1])

            received_msg = b''
            received_msg_status = True
            iteration = iteration + 1

        elif iteration == 2 and len(received_msg)-HEADERSIZE == recived_msglen:
            nparr = np.frombuffer(received_msg[HEADERSIZE:], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            print("Received frame: ", receivedCount)

            received_msg = b''
            received_msg_status = True

            images.append(img)
            receivedCount = receivedCount + 1
            if receivedCount == limit:
                received_msg = b''
                received_msg_status = True
                iteration = iteration + 1

        elif iteration == 3 and len(received_msg)-HEADERSIZE == recived_msglen:
            stat = received_msg[HEADERSIZE:].decode('utf8')

            received_msg = b''
            received_msg_status = True
            break
    break

s.close()
print('Total recv: ', len(images))

incoming_stat = stat.split(',')
sensor_time = float(incoming_stat[0])
sensor_time2 = float(incoming_stat[1])
sensor_mem = float(incoming_stat[2])
sensor_size = float(incoming_stat[3])

if len(images) == limit:
    print("clear to process")
    process_image()
else:
    print("Frames have not arrived")
    s.close()
    os._exit(0)
