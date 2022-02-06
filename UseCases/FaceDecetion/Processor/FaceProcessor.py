import cv2
import statistics
import glob
import csv
import numpy as np
import time
from time import process_time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import socket
import sys
import os
import tracemalloc

print("Starting the processor")
############################################## Knob settings ############################################
_scaleFactor = float(sys.argv[1])
a = int(sys.argv[2])
_minSize = (a, a)

need_Reconstruct = int(sys.argv[3])
# Contrast limited adaptive histogram equalization.
hist_eq = int(sys.argv[4])

config_param = 'Scale:_'+str(_scaleFactor)+'_Window:_'+str(_minSize)+'_Filter?:_' + \
    str(need_Reconstruct)+'_Contrast?:_'+str(hist_eq)+'_'

if need_Reconstruct == True:
    filter_param = str(sys.argv[5])
    filter_arg = int(sys.argv[6])

    config_param = config_param+'FiltType:_' + \
        filter_param+'_FiltParam:_'+str(filter_arg)
print(config_param)
############################################## Knob settings ############################################

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
# clipLimit can be used to adjust contrast.
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(1, 1))
str1 = './Face_Truth/'  # ground truth
str2 = './Face_Test/'  # received faces from sensor
############################################## File Cleaning ############################################
files = glob.glob('./Face_Test/*.png')
if len(files) > 0:
    for f in files:
        os.remove(f)
############################################## File Cleaning ############################################
print("Ready to connect")

images = []
receivedCount = 0

HEADERSIZE = 10
host = ''  # host ip address
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

connected = False
while not connected:
    try:
        s.connect((host, 12345))
        connected = True
    except Exception as e:
        print(e)  # Do nothing, just try again
print("Connection established")


def csvWriter(sen_param, proc_param, ProWallTime, ProCPUTime, ProMem, ProPSNR, ProSSIM, SenWallTime, SenCPUTime, SenMem, TransVol, GlobalWallTime, GlobalCPUTime, GlobalMem):
    with open("Stat.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)

        ret = writer.writerow(
            [sen_param, proc_param, ProWallTime, ProCPUTime, ProMem, ProSSIM, SenWallTime, SenCPUTime, SenMem, TransVol, GlobalWallTime, GlobalCPUTime, GlobalMem])
        return ret


def compare_face(str1, str2):
    count = 0
    scoreSetImageMean_ssim = []
    scoreSetImageMean_psnr = []
    mean_ssim = 0
    mean_psnr = 0

    while count < limit:
        imageSet1 = []
        imageSet2 = []
        pattern = 'Face_'+str(count)+'_*.png'
        count = count+1
        imageSet1 = [cv2.imread(file) for file in glob.glob(str1+pattern)]
        imageSet2 = [cv2.imread(file) for file in glob.glob(str2+pattern)]

        scoreSetFrame_ssim = []
        scoreSetFrame_mse = []
        for frame1 in imageSet1:
            grayA = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            max_score_ssim = 0
            max_score_mse = 0
            for frame2 in imageSet2:
                grayB = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                if grayA.shape[1] != grayB.shape[1] and grayA.shape[0] != grayB.shape[0]:
                    dim = (grayA.shape[1], grayA.shape[0])
                    grayB = cv2.resize(
                        grayB, dim, interpolation=cv2.INTER_CUBIC)

                try:
                    score_ssim = ssim(grayA, grayB)
                    score_mse = psnr(grayA, grayB)
                    if score_ssim > max_score_ssim:
                        max_score_ssim = score_ssim
                    if score_mse > max_score_mse:
                        max_score_mse = score_mse
                except ValueError:
                    pass

            # Sometimes these 2 lists are becoming null
            scoreSetFrame_ssim.append(max_score_ssim)
            scoreSetFrame_mse.append(max_score_mse)

        try:
            scoreSetImageMean_ssim.append(statistics.mean(scoreSetFrame_ssim))
            scoreSetImageMean_psnr.append(statistics.mean(scoreSetFrame_mse))
        except statistics.StatisticsError:
            pass

    try:
        mean_ssim = statistics.mean(scoreSetImageMean_ssim)
        mean_psnr = statistics.mean(scoreSetImageMean_psnr)
    except statistics.StatisticsError:
        pass

    return mean_ssim, mean_psnr


def image_reconstructor(image):
    if filter_param == 'non':
        dst = cv2.fastNlMeansDenoising(image, 10, 21, filter_arg)

    elif filter_param == 'gaus':
        dst = cv2.GaussianBlur(image, (filter_arg, filter_arg), 0)

    if hist_eq == True:
        dst = clahe.apply(dst)
    return dst


def face_Detector(frame, frameCount):
    faceCount = 0
    status = False
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=_scaleFactor,  minSize=_minSize)

    if len(faces) != 0:
        for x, y, w, h, in faces:
            faceimg = frame[y:y+h, x:x+w]
            status = cv2.imwrite('./Face_Test/Face_'+str(frameCount) +
                                 '_'+str(faceCount)+'.png', faceimg)
            faceCount = faceCount + 1
    else:
        print("No face detected ")
        faceCount = faceCount + 1
        status = True

    time.sleep(.3)
    return status


def process_image():
    frameCount = 0
    for img in images:
        if need_Reconstruct == True:
            if np.all(img == 0):
                print('Got empty one')
                frameCount = frameCount+1
                continue
            else:
                img_Reconstructed = image_reconstructor(img)
                status = face_Detector(img_Reconstructed, frameCount)
                if status is True:
                    print("Processed frame: ", frameCount)
                    frameCount = frameCount+1
        else:
            if np.all(img == 0):
                frameCount = frameCount+1
                continue
            else:
                status = face_Detector(img, frameCount)
                if status is True:
                    print("Processed frame: ", frameCount)
                    frameCount = frameCount+1

    if limit == frameCount:
        print("Now")
        ########## Stop the stopwatch / counter#########
        t1_stop = process_time()
        passed = round((t1_stop-t1_start), 2)
        end = time.time()
        elapsed = round((end - start), 2)
        _, first_peak = tracemalloc.get_traced_memory()
        memoryUSed = round((first_peak/10**6), 2)

        meanFrameScore_ssim1, meanFrameScore_psnr1 = compare_face(str1, str2)
        meanFrameScore_ssim2, meanFrameScore_psnr2 = compare_face(str2, str1)
        ssim_face = round(
            ((meanFrameScore_ssim1 + meanFrameScore_ssim2)/2), 2)
        psnr_face = round(
            ((meanFrameScore_psnr1 + meanFrameScore_psnr2)/2), 2)

        # ssim_img, mse_img = compare_img(strA, strB)

        ret = csvWriter(incoming_param, config_param, elapsed, passed, memoryUSed, psnr_face, ssim_face*100, sensor_time,
                        sensor_time2, sensor_mem, sensor_size, elapsed+sensor_time, passed+sensor_time2, memoryUSed+sensor_mem)
        if ret > 0:
            s.close()
            os._exit(0)


terminate_counter = 0
while True:
    received_msg = b''
    received_msg_status = True
    iteration = 1

    while True:
        terminate_counter = terminate_counter+1
        if terminate_counter > 15000:
            print("Frames will not arrived")
            tracemalloc.reset_peak()
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
    tracemalloc.reset_peak()
    s.close()
    os._exit(0)
