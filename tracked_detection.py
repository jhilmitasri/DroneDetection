import torch
import cv2
import torchvision.transforms as T
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import numpy as np
import os

def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 states, 2 measurements

    # State Transition Matrix
    dt = 1.0  # time step
    kf.F = np.array([[1, 0, dt, 0], 
                     [0, 1, 0, dt], 
                     [0, 0, 1, 0], 
                     [0, 0, 0, 1]])

    # Measurement Function
    kf.H = np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0]])

    # Initial State Estimate
    kf.x = np.zeros(4)

    # Covariance Matrix
    kf.P *= 1000.  # initial uncertainty

    # Process Noise
    kf.Q = np.eye(4) * 0.1

    # Measurement Noise
    kf.R = np.eye(2) * 5

    return kf

def update_kalman_filter(kf, detection):
    kf.predict()
    kf.update(detection)
    return kf.x

# Initialize Kalman Filter
kf = initialize_kalman_filter()

# Load the model
model = YOLO('train/weights/best.pt')
# model.eval()

# Define transformation
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((640, 640)),
    T.ToTensor(),
])

# Function to draw trajectory
def draw_trajectory(frame, trajectory_points):
    for i in range(len(trajectory_points) - 1):
        cv2.line(frame, trajectory_points[i], trajectory_points[i + 1], (0, 0, 0), 2)
    return frame

# Load videos from a directory
video_dir = 'input_videos'
output_dir = 'detections'
os.makedirs(output_dir, exist_ok=True)

for video_file in os.listdir(video_dir):
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define output video writer for frames with detections
    output_video_path = os.path.join(output_dir, f'detected_{video_file}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 1, (frame_width, frame_height))  # 1 fps
    
    frame_time = 0  # in milliseconds
    trajectory_points = []

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
        frame_time += 100  # increment by 1000 ms for 1 fps
        

        ret, frame = cap.read()
        if not ret:
            break
        y_ = frame.shape[0]
        x_ = frame.shape[1]

        targetSize = 640
        x_scale = x_ / targetSize
        y_scale = y_ / targetSize
        print("Image Size: ", frame.shape)
        img = transform(frame).unsqueeze(0)
        output_frame = cv2.resize(frame, (targetSize, targetSize))
        with torch.no_grad():
            results = model(img)

        detection = None
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2, y1, y2 = int(np.round(x1*x_scale)), int(np.round(x2*x_scale)), int(np.round(y1*y_scale)), int(np.round(y2*y_scale))
                confidence = box.conf[0]
                label = int(box.cls[0])
                print("x1, y1, x2, y2", x1, y1, x2, y2, "Conf: ", confidence, "Label: ", label)
                # if confidence > 0.6 and label == 0:
                detection = [(x1 + x2) / 2, (y1 + y2) / 2]  # center of the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 2)
                # cv2.putText(frame, f'Drone: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(output_dir, f'detection_{frame_time}.jpg'), frame)  # save frame with detection

        if detection:
            kf_state = update_kalman_filter(kf, detection)
            trajectory_points.append((int(kf_state[0]), int(kf_state[1])))

        # Draw trajectory on the frame
        frame = draw_trajectory(frame, trajectory_points)

        # Write frame to output video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press 'q' on the keyboard to exit the display window early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()

cv2.destroyAllWindows()
