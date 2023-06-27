import cv2
import numpy as np

dispW = 640
dispH = 480

font = cv2.FONT_HERSHEY_SIMPLEX

# Set evt & coord variables
evt = -1
coord = []

def click(event, x, y, flags, params):
    global pt
    global evt
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Mouse Event Was: ', event)
        print(x, ',', y)
        pt = (x,y)
        coord.append(pt)
        evt = event
        
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', click)

cam = cv2.VideoCapture('vidPengujian/Pengujian_livingRoom_Edward.mp4')

while True:
    ret, frame = cam.read()

    for pnts in coord:
        cv2.circle(frame, pnts, 3, (0,0,255))
        myStr = str(pnts)
        cv2.putText(frame, myStr, pnts, font, 0.6, (255,0,0), 2)
    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', 0,0)
    
    if cv2.waitKey(20) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()