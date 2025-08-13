import cv2
import numpy as np
import os
import sys

# Load class labels MobileNet SSD was trained on (COCO dataset 21 classes)
classNames = { 0: 'background',
               1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
               5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
               9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
               13: 'horse', 14: 'motorbike', 15: 'person',
               16: 'pottedplant', 17: 'sheep', 18: 'sofa',
               19: 'train', 20: 'tvmonitor' }

# Load the pre-trained model files from the local 'models' directory
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
modelFile = os.path.join(models_dir, "MobileNetSSD_deploy.caffemodel")
configFile = os.path.join(models_dir, "MobileNetSSD_deploy.prototxt")

# if the files are not found, declare an error & stop
if not os.path.isfile(configFile) or not os.path.isfile(modelFile):
    print("Error: Model files not found.")
    print(f"Expected prototxt at: {configFile}")
    print(f"Expected caffemodel at: {modelFile}")
    sys.exit(1)

# Initialize mobilenet
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Initialize video capture from webcam (index 0)
cap = cv2.VideoCapture(0)

# Constant object detection loop
while True:
    ret, frame = cap.read() # read webcam frame
    if not ret: # if no frame returned
        break

    # Get frame dimensions
    h, w = frame.shape[:2]

    # Prepare input blob and perform a forward pass to get detections
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        # get confidence for ith detection
        confidence = detections[0, 0, i, 2] # 0 = 1 image @ 0, 0 = 1st output, i = ith detection, 2 = confidence

        # Ignore weak detections
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])  # Get the detection index
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # upscale detection fractions
            (startX, startY, endX, endY) = box.astype("int") # convert all to ints 

            label = "{}: {:.2f}%".format(classNames.get(idx, "Unknown"), confidence * 100) # detection display label
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2) # draw rectangle based on detection bounds
            y = startY - 15 if startY - 15 > 15 else startY + 15 # above box by 15, else below box by 15 (px)
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show output frame
    cv2.imshow("MobileNet SSD - Live Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup when q pressed
cap.release()
cv2.destroyAllWindows()
