import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture("vidi.mp4") #reading video
video_width=cap.get(3)
video_height=cap.get(4)

human_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')#apply haarcascade algorithm

while True:
    _, frame = cap.read()#get 2 value _ and frame
    frame=imutils.resize(frame,width=10000) #resize your video this is important for get better results
    griton = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #change the video withgray color
    people=human_cascade.detectMultiScale(griton, 1.1, 4, minsize=(400,400), flags=cv2.CASCADE_SCALE_IMAGE)
    #you have to change thiss part up to your video especially scalefactor and minsize

    for (x, y, w, h) in people: #putting rectangles on detected bodies
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)
    cv2.imshow('Body', frame) #showing the output video
    if cv2.waitKey(25) & 0xFF == ord('q'): #press q for quit
        break

cap.release()
cv2.destroyAllWindows()
