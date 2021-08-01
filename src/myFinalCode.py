from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from winsound import PlaySound, SND_FILENAME, SND_LOOP, SND_ASYNC
import argparse
import imutils
import time
import dlib
import cv2
#import playsound

def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
   
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar 

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,help="index of webcam on system")
args = vars(ap.parse_args())
 

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 25
MOUTH_AR_THRESH = 0.32
MOUTH_AR_CONSEC_FRAMES = 25
YAWN_THRESHOLD = 4
YAWN_COUNTER = 0
YAWN_TIME_COUNTER = 0
curr_yawn = 0
curr_yawn_for_eye=0
EYE_TIME_COUNTER = 0
ALARM_ON = False
flag = 0
ALARM_TIME_COUNT = 0
ALARM_TIME_THRESHOLD = 40

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
       
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        MAR = mouth_aspect_ratio(mouth)
        EAR = (leftEAR + rightEAR) / 2.0

       
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (255, 255, 255), 1)
       # for (x, y) in shape[lStart:lEnd]:
       #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
        
        if flag == 2:
            ALARM_TIME_COUNT += 1
            cv2.putText(frame, "YAWNING ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if flag == 1:
             cv2.putText(frame, "SLEEPING ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
     

        if EAR < EYE_AR_THRESH:
          if curr_yawn_for_eye != 1:  
            EYE_TIME_COUNTER += 1

            if EYE_TIME_COUNTER >= EYE_AR_CONSEC_FRAMES:
              if flag == 0 : 
                if not ALARM_ON:
                    ALARM_ON = True
                    flag=1
                    PlaySound('Fire_Alarm.wav',SND_FILENAME|SND_LOOP|SND_ASYNC)         
 #               cv2.putText(frame, "SLEEPING ALERT!", (10, 30),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
          if flag==1:
            EYE_TIME_COUNTER = 0
            ALARM_ON = False
            flag = 0
            PlaySound(None, SND_FILENAME)
        
        if MAR > MOUTH_AR_THRESH:
            curr_yawn_for_eye = 1
            YAWN_TIME_COUNTER += 1
 
            if YAWN_TIME_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
             if curr_yawn==0:  
              YAWN_COUNTER +=1
              curr_yawn = 1;
              YAWN_TIME_COUNTER = 0
              if YAWN_COUNTER>=YAWN_THRESHOLD:
                if not ALARM_ON:
                    ALARM_ON = True
                    flag=2
                    PlaySound('Fire_Alarm.wav',SND_FILENAME|SND_LOOP|SND_ASYNC)
 #               cv2.putText(frame, "YAWNING ALERT!", (10, 30),
 #                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
           curr_yawn_for_eye = 0 
           curr_yawn = 0
           if flag==2:
             if ALARM_TIME_COUNT > ALARM_TIME_THRESHOLD:
                ALARM_ON = False
                ALARM_TIME_COUNT = 0
                flag = 0
                YAWN_COUNTER  = 0
                PlaySound(None, SND_FILENAME) 
        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (360, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(MAR), (360, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN COUNT: " + str(YAWN_COUNTER), (360, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
  
    cv2.imshow("DROWSINESS DETECTOR", frame)
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()