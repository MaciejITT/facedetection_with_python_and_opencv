import cv2
import keyboard


trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# x value represents your camera (default should be 0)
# you can also search faces in videos by giving path under x egz.: 'egz_vid.mp4'
x = 2
web_cam = cv2.VideoCapture(x)

while True:
    successful_frame_read, frame = web_cam.read()

    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 255, 0), 2)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113 or key == 27:
        break

web_cam.release()
