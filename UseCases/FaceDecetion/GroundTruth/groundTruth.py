from cv2 import cv2
import glob
import os

############################################## File Cleaning ############################################
# ground truth storage in processor device.
files = glob.glob('./Face_Truth/*.png')
if len(files) > 0:
    for f in files:
        os.remove(f)
############################################## File Cleaning ############################################

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")


def sortKeyFunc(s):
    return int(os.path.basename(s)[5:-4])


images = [cv2.imread(file) for file in sorted(
    glob.glob("./TestDataDown/*.jpg"), key=sortKeyFunc)]  # your local image dataset

# video = cv2.VideoCapture("./samples/sample4.mp4")
# video = cv2.VideoCapture(0)

############################################## Alg. Knobs ##############################################

# # The argument scaleFactor determines the factor by which the detection window of the classifier
# is scaled down per detection pass. A factor of 1.1 corresponds to an increase of
# 10%. Hence, increasing the scale factor increases performance, as the number of detection passes is
# reduced. However, as a consequence the reliability by which a face is detected is reduced. You may
# increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether.
_scaleFactor = 1.005
# # The argument minNeighbor determines the minimum number of neighboring facial features that need to
# be present to indicate the detection of a face by the classifier. Decreasing the factor increases the
# amount of false positive detections. Increasing the factor might lead to missing faces in the image.
# The argument seems to have no influence on the performance of the algorithm. 3~6 is a good value for it.
# _minNeighbors = 4
# # The argument minSize determines the minimum size of the detection window in pixels. Increasing the
# minimum detection window increases performance. However, smaller faces are going to be missed then.
_minSize = (10, 10)
############################################## Knobs ##############################################


def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname, img)

########################## 2nd phase of the program ##################################


def process(images):
    count1 = 0
    for frame in images:
        count2 = 0
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img,
                                              scaleFactor=_scaleFactor, minSize=_minSize)

        if len(faces) != 0:
            for x, y, w, h, in faces:
                faceimg = gray_img[y:y+h, x:x+w]
                cv2.imwrite('./Face_Truth/Face_'+str(count1) +
                            '_'+str(count2)+'.png', faceimg)
                count2 = count2+1
        else:
            print("No face detected for face: ",
                  './Face_Truth/Face_'+str(count1))
            count2 = count2+1

        count1 = count1+1


process(images)
del images
############################ For video ################################
# while True:
#     check, frame = video.read()
#     if check == True:
#         processing...
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#     else:
#         print("The end  ")
#         break
# video.release()
############################ For video ################################
