import cv2
import time
import math


video = cv2.VideoCapture("bb3.mp4")

tracker= cv2.TrackerCSRT_create()

returned, img= video.read()

bbox= cv2.selectROI("Tracking", img, False)

tracker.init(img, bbox)


print(bbox)
#Tuple
#bbox= (250,120,70,60)
#bbox[2]
def drawBox(img, bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img, (x,y), ((x+w),(y+h)),  (250,0,255),3,1 )
    cv2.putText(img,"Tracking", (75,90),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.7,(0,255,0),2)


while True:
    
    check, img = video.read()   

    # Update the tracker on the img and the bounding box
    success, bbox=  tracker.update(img)

    if success: 
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost the object", (75,90), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7,(255,0,0),2)



    # Display Video
    cv2.imshow("result", img)


    # Quit Display Window when Spacebar key is pressed        
    key = cv2.waitKey(25)
    if key == 32:
        print("Stopped")
        break

video.release()
cv2.destroyAllWindows()