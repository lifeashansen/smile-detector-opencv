import os

import cv2

os.environ["QT_QPA_PLATFORM"] = "xcb"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, flipped_frame = cap.read()

    if not ret:
        print("Failed to grab frames")
        break

    frame = cv2.flip(flipped_frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.3)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]

        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25)
        )

        for sx, sy, sw, sh in smiles:
            cv2.rectangle(
                frame,
                (x + sx, y + sy),
                (x + sx + sw, y + sy + sh),
                (0, 255, 0),
                2,
            )

    cv2.imshow("I cant see you :3", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
