import cv2
import sys
import numpy as np
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    params = dict( )
    
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']
    tracker_type = tracker_types[2]
    tracker = cv2.TrackerKCF.create(params = params)
#    tracker = cv2.TrackerKCF.create(params = params)
#    tracker = cv2.TrackerKCF_create(params = params)
#    if tracker_type == 'BOOSTING':
#        tracker = cv2.TrackerBoosting_create()
#    if tracker_type == 'MIL':
#        tracker = cv2.TrackerMIL_create()
#    if tracker_type == 'KCF':
#        tracker = cv2.TrackerKCF_create()
#    if tracker_type == 'TLD':
#        tracker = cv2.TrackerTLD_create()
#    if tracker_type == 'MEDIANFLOW':
#        tracker = cv2.TrackerMedianFlow_create()
#    if tracker_type == 'GOTURN':
#        tracker = cv2.TrackerGOTURN_create()
#    if tracker_type == 'MOSSE':
#        tracker = cv2.TrackerMOSSE_create()
        
    # Read video
    video = cv2.VideoCapture(r"D:\CV_Library\MyTestVideo_1.mp4")
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)
    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    
    enter_ascii = 13
    
    frame_num = video.get(7)
    frame_round = frame_num / 100 -1
    
    break_flag = True
    for i in range(int(frame_round)):
        if break_flag == False:
            break
        a = np.random.rand(1)*frame_round
        print("随机起始帧：%d"%np.int(np.floor(a*100)))
        video.set(cv2.CAP_PROP_POS_FRAMES,np.int(np.floor(a*100)))
        count = 100
        while count>0:
            ok, frame = video.read()
            if not ok:
                break
            # Start timer
            timer = cv2.getTickCount()
            # Update tracker
            ok, bbox = tracker.update(frame)
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
            # Display result
            cv2.imshow("Tracking", frame)
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 13 : break
        cv2.destroyAllWindows()