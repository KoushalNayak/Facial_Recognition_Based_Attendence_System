import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import csv

# Load dataset with subfolders and multiple images per person
def load_known_faces(dataset_path="dataset"):
    known_face_encodings = []
    known_face_names = []
    known_face_usns = []  # New list to store USNs

    print(f"Loading dataset from {dataset_path}...")

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                try:
                    image_path = os.path.join(root, file)
                    image = face_recognition.load_image_file(image_path)

                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
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

    return known_face_encodings, known_face_names, known_face_usns

# Initialize or update attendance file
def initialize_attendance_file(output_file="attendance.csv"):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "USN", "Status", "Time"])
    print(f"Attendance file initialized: {output_file}")

# Update the attendance record in the file
def update_attendance_file(name, usn, status, output_file="attendance.csv"):
    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        now = datetime.now().strftime("%H:%M:%S")
        writer.writerow([name, usn, status, now])
    print(f"Attendance updated: {name} ({usn}), {status}, {now}")

# Main function
def main():
    print("Loading known faces...")
    known_face_encodings, known_face_names, known_face_usns = load_known_faces()

    if not known_face_names:
        print("No faces found in the dataset. Please check your dataset.")
        return

    print(f"Loaded {len(known_face_names)} faces.")

    # Initialize attendance dictionary and CSV file
    attendance_dict = {name: "Absent" for name in known_face_names}
    initialize_attendance_file()

    # Open webcam
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error accessing webcam.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
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
                        print(f"Recognized and marked present: {name} ({usn})")

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    print("Attendance process completed. Check attendance.csv for results.")

if __name__ == "__main__":
    main()
