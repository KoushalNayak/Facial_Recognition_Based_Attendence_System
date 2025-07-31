from flask import Flask, render_template, Response, redirect, url_for, send_file, request
import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime
import threading

# Initialize Flask app
app = Flask(__name__)

# Directory for dataset and attendance file
DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"

# Load known faces
known_face_encodings = []
known_face_names = []
known_face_usns = []

# Shared resources for frame processing
frame_queue = []
lock = threading.Lock()


def load_known_faces():
    global known_face_encodings, known_face_names, known_face_usns
    print(f"Loading dataset from {DATASET_DIR}...")
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                try:
                    image_path = os.path.join(root, file)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        face_encoding = encodings[0]
                        folder_name = os.path.basename(root)

                        # Assume folder name is in "Name_USN" format
                        if "_" in folder_name:
                            name, usn = folder_name.split("_", 1)
                        else:
                            name, usn = folder_name, "Unknown USN"

                        known_face_encodings.append(face_encoding)
                        known_face_names.append(name)
                        known_face_usns.append(usn)
                        print(f"Loaded: {name} ({usn}) from {file}")
                    else:
                        print(f"No face found in {file}, skipping...")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")


# Initialize attendance CSV
def initialize_attendance_file():
    with open(ATTENDANCE_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "USN", "Status", "Time"])


# Update attendance
def update_attendance_file(name, usn, status):
    with open(ATTENDANCE_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        now = datetime.now().strftime("%H:%M:%S")
        writer.writerow([name, usn, status, now])


def generate_frames():
    """Generates video frames for the webcam feed."""
    print("Starting video capture...")
    video_capture = cv2.VideoCapture(0)
    attendance_dict = {name: "Absent" for name in known_face_names}
    process_this_frame = True

    if not video_capture.isOpened():
        print("Error: Webcam not accessible!")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Frame not captured!")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                name, usn = "Unknown", "Unknown USN"

                if True in matches:
                    match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                    name = known_face_names[match_index]
                    usn = known_face_usns[match_index]

                    if attendance_dict[name] == "Absent":
                        attendance_dict[name] = "Present"
                        update_attendance_file(name, usn, "Present")

        process_this_frame = not process_this_frame

        # Draw rectangles and labels
        for (top, right, bottom, left), name in zip(face_locations, [name] * len(face_locations)):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Encode the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()
    print("Video capture ended.")


# Flask routes
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/start-webcam')
def start_webcam():
    return render_template('webcam.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result():
    return send_file(ATTENDANCE_FILE, as_attachment=True)


@app.route('/quit')
def quit_app():
    return redirect(url_for('home'))


@app.route('/revert', methods=['POST'])
def revert():
    initialize_attendance_file()  # Reinitialize the attendance file
    return redirect(url_for('home'))


# Main block
if __name__ == "__main__":
    load_known_faces()
    initialize_attendance_file()
    app.run(debug=True)
