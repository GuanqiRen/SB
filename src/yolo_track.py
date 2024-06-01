import cv2
from ultralytics import YOLO
import json, os

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "/workspace/hpe/SB/data/GX012146.mp4"
video_out ="output.mp4"
json_out = '/workspace/hpe/SB/data/res.json'
cap = cv2.VideoCapture(video_path)
if os.path.isfile(json_out):
    os.remove(json_out)
f = open(json_out, 'a+')
if os.path.isfile(video_out):
    os.remove(video_out)
output = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (3840, 2160))
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        json.dump(results[0].summary(), f)
        f.write('\n')
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #import pdb;pdb.set_trace()
        output.write(annotated_frame)
        #import pdb;pdb.set_trace()
        # Display the annotated frame
        #cv2.imwrite('test.jpg', annotated_frame)
        #cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break
    else:
        # Break the loop if the end of the video is reached
        break
f.close()
# Release the video capture object and close the display window
cap.release()
output.release() 
cv2.destroyAllWindows()

