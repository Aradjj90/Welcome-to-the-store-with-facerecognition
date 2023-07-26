import face_recognition
import cv2
import numpy as np
import os
import pyttsx3

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6 #Recognition accuracy, 0 = 100% correlation
MODEL = 'hog'

engine = pyttsx3.init()
voices = engine.getProperty('voices')

def speak(audio):
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 135)  # setting up new voice rate
    engine.say(audio)
    engine.runAndWait()

print('Loading known faces...')
known_faces = []
known_names = []
face_names = []
# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

video_capture = cv2.VideoCapture(1)
video_capture.set(3, 680)  # set Width
video_capture.set(4, 420)  # set Height
brightness = 50
contrast = 30

process_this_frame = True

while(True):
    ret, image = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    image = np.int16(image)
    image = image * (contrast / 127 + 1) - contrast + brightness
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    gray_frame = image[:, :, ::-1]
    if process_this_frame:
        locations = face_recognition.face_locations(gray_frame, model=MODEL)
        encodings = face_recognition.face_encodings(gray_frame, locations)
        face_names = []
        for face_encoding in encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            person = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     person = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                person = known_names[best_match_index]
            if "Unknown" in person:
                speak("welcome unknown person")
            else:
                speak("hello, %s" % person)
            face_names.append(person)

    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), person in zip(locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        # top *= 2
        # right *= 2
        # bottom *= 2
        # left *= 2

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 22), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, person, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Show image
    cv2.imshow("Video", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
