import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load YOLO model
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


def detect_people(image):
    """Detect people in the image using YOLO and return the number of persons detected."""
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(
        image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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


@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint to detect people in a base64 image."""
    try:
        data = request.json
        # Extract the base64 encoded image string
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        # Decode the base64 image
        image_data = base64.b64decode(image_base64)
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Detect people in the image
        num_people = detect_people(image)
        print(num_people)

        # Return 0 if no person or more than one person is detected, else 1
        result = 1 if num_people == 1 else 0
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
