import numpy as np
import cv2

cap = cv2.VideoCapture('workspace/img_dir/tst.mp4')

fourcc = cv2.VideoWriter_fourcc(*'H264')

ret, frame = cap.read()
vid_size = frame.shape[:2][::-1]

out = cv2.VideoWriter('workspace/img_dir/testwrite.mp4',fourcc, 8, vid_size)
out.write(frame)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret: break
    out.write(frame)
        

cap.release()
out.release()



