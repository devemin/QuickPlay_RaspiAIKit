import cv2

#src = 'videotestsrc ! videoconvert ! appsink'
src = 'libcamerasrc name=src_0 auto-focus-mode=2 ! queue name=queue_src_scale max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! videoscale ! video/x-raw,format=RGB,width=1024,height=768,framerate=30/1 ! videoconvert ! videoscale ! appsink'


cap = cv2.VideoCapture(src)

while(cap.isOpened()):
    ret, frame = cap.read()

    if frame is None:
        break
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()