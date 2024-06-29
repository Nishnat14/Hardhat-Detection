import streamlit as st
import cv2
import math
import cvzone
from ultralytics import YOLO
import tempfile
import os

st.title("Helmet Detection System")

# Initialize YOLO model
model = YOLO("finalhardhat.pt")
classNames = ['Hardhat', 'NO-Hardhat']
myColor = (0, 0, 255)

# Function to process the video stream and display detection
def process_video_stream(video_file=None):
    if video_file is None:
        cap = cv2.VideoCapture(0)  # For Webcam
        cap.set(3, 1280)
        cap.set(4, 720)
    else:
        # Save the uploaded video file temporarily
        temp_video_path = "uploaded_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        # Open the temporarily saved video file
        cap = cv2.VideoCapture(temp_video_path)  # For Uploaded Video

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_file_path = "output.avi"
    out = cv2.VideoWriter(temp_file_path, fourcc, 30.0, (640, 480))

    # Create a section for video streaming
    video_section = st.empty()

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                if conf > 0.5:
                    if currentClass == 'NO-Hardhat':
                        myColor = (0, 0, 255)
                    if currentClass == 'Hardhat':
                        myColor = (0, 255, 0)

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        # Write the frame into the file 'output.avi'
        out.write(img)

        # Display the video stream in the specified section
        video_section.image(img, channels="BGR", use_column_width=True)


    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
# Streamlit UI
st.write("Choose an option to start object detection.")
option = st.radio("Select Option", ["Webcam", "Upload Video"])

if option == "Webcam":
    st.write("Click 'Start' to begin object detection from webcam stream.")
    if st.button('Start'):
        process_video_stream()

elif option == "Upload Video":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        st.write("Click 'Start' to begin object detection on the uploaded video.")
        if st.button('Start'):
            process_video_stream(video_file)