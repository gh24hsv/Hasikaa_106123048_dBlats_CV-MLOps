import cv2 # type: ignore

from util import get_parking_spots_bboxes, empty_or_not, calc_diff # Utils file contains the important functions we use for classification
import numpy as np #type: ignore

video_path = './vids/parking_loop.mp4'

mask = './mask.png'

cv2.imread(mask, 0) # Open image as grayscale, we use this mask to get bounding boxes of all the parking spots

cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(cv2.imread(mask, 0), 4, cv2.CV_32S) # gets connected components aka parking spots in mask img

spots = get_parking_spots_bboxes(connected_components)

#print(spots[0])

spots_status = [None for j in spots] # Stores whether spot empty or not for each spot in a frame
diffs = [None for j in spots] # Stores the difference between spot in curr and prev frame (the func uses np.mean)

prev_frame = None 

ret = True # return value of whether or not frame is succesfully read

step = 30 # parking a car generally takes 5-10 seconds so skipping prediction for
# 'step' number of frames will help in optimizing for real-time classification of really large parking lots 

frame_count = 0

while ret:
    ret, frame = cap.read()

    if frame_count % step == 0 and prev_frame is not None:
        for spot_idx, spot in enumerate(spots):
            (x1, y1, w, h) = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :] # gets the crop of particular iter spot

            diffs[spot_idx] = calc_diff(spot_crop, prev_frame[y1:y1 + h, x1:x1 + w, :]) # calculates the diff between mean of the spot in prev and curr frame

    if frame_count % step == 0:
        if prev_frame is None:
            arr = range(len(spots)) # Process all spots, arr = [0,1,2,...]
        else:
            arr = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4] # Process only those spots where the scaled cal_diff > 0.4
        for spot_idx in arr:
            (x1, y1, w, h) = spots[spot_idx]

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            spot_status = empty_or_not(spot_crop)
            spots_status[spot_idx] = spot_status
    
    if frame_count % step == 0:
        prev_frame = frame.copy() # update prev frame

    for spot_idx, spot in enumerate(spots):
        x1, y1, w, h = spots[spot_idx]
        spot_status = spots_status[spot_idx]
        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2) # draw green rect if empty

            # dot_x, dot_y = x1 - 10, y1 + h // 2  # Position the dot on the left
            # dot_x1, dot_y1 = x1 + w + 10, y1 + h // 2  # Position the dot on the right
            # cv2.circle(frame, (dot_x, dot_y), 5, (0, 255, 0), -1)
            # cv2.circle(frame, (dot_x1, dot_y1), 5, (0, 255, 0), -1)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2) # draw red rect if occupied

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL) # create a window to diplay the frame
    cv2.imshow('frame', frame) # showing the frame with markings 
    if cv2.waitKey(25) & 0xFF == ord('q'): # exit loop is 'q' is pressed 
        break

    frame_count += 1


cap.release()
cv2.destroyAllWindows()

