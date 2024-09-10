import cv2
import numpy as np

# Load YOLO
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
labels_path = "coco.names"

# Load the YOLO network
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(labels_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]


def detect_people(frame):
    """Detect people in the image frame using YOLO and return the number of persons detected."""
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            # Ensure the detection has at least 5 elements
            if len(detection) > 5:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Check if it's a 'person' class and confidence is high enough
                if confidence > 0.5 and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    num_people = len(indexes) if indexes is not None else 0

    return num_people


def main():
    """Capture video from webcam and detect the number of people."""
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection and get the number of people
        num_people = detect_people(frame)

        # Print the result in the terminal
        print(f"Number of people detected: {num_people}")

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()


if __name__ == "__main__":
    main()
