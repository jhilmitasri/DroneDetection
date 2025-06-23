import torch
import cv2
import torchvision.transforms as T
from ultralytics import YOLO

# Load the model
model = YOLO('train/weights/best.pt')
# model.eval()

# Define transformation
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((640, 640)),
    T.ToTensor(),
])

# Load video
video_path = 'input_videos/drone_video_1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video details
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
output_path = 'op_drone_video_1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 1, (frame_width, frame_height))  # 1 fps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Transform the frame
    img = transform(frame).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        results = model(img)

    # Process predictions
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = int(box.cls[0])

            if confidence > 0.5 and label == 0:  # Assuming 'drone' has label 0
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Drone: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write frame to output video
    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' on the keyboard to exit the display window early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
