import face_recognition
import cv2
import os
import time

def draw_rectangle(frame, left, right, bottom, top):
    #Draws a bounding box around identified face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 25), (right, bottom ), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left+ 6, bottom - 6), font, 0.5, (255,255,255), 1)
    return

camera = cv2.VideoCapture(0)
time.sleep(1)
ret,reference_frame = camera.read() # returns a single frame as reference

cv2.imshow('Reference image',reference_frame) # display the captured image
cv2.waitKey(0)

ref_face_encodings = [faces for faces in face_recognition.face_encodings(reference_frame)]

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    #Grab single frame from video
    ret, frame = camera.read()
    frame_number += 1

    # Convert from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = frame[:,:,::-1]
    # Find faces and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame,face_locations)

    face_names = []

    for  face_encodings in face_encodings:
        # Match the detected faces with known faces
        match = face_recognition.compare_faces(ref_face_encodings,face_encodings,tolerance = 0.50)

        name = None
        for i in range(len(match)):
            if match[i]:
                name = "Person " + str(i + 1)

            face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations,face_names):
                if not name:
                    continue
                
                draw_rectangle(frame, left, right, bottom, top)

    cv2.imshow("Live",frame)

    k = cv2.waitKey(1) & 0xFF   #0xFF is hexadecimal for 11111111 in binary

    if k == ord('q'):       #ord() returns integer representing unicode point for 'q'
        break

camera.release()
cv2.destroyAllWindows()

