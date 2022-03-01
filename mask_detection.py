from keras.models import load_model
import cv2
import numpy as np
import tkinter
from tkinter import messagebox
import smtplib

# initialize tkinter
root = tkinter.Tk()
root.withdraw()

# load model
model = load_model('face_mask_detection.h5')

# face classifier
face_det_classifier = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

vid_source = cv2.VideoCapture(0)
text_dict = {0: 'Mask ON', 1: 'No Mask'}
rect_color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

Subject = 'Subject'
Text = 'A person had been detected without a face mask'

while True:
    ret, img = vid_source.read()
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_det_classifier.detectMultiScale(grayscale_img, 1.3, 5)

    for(x, y, w, h) in faces:
        face_img = grayscale_img[y:y+w, x:x+w]
        resized_img = cv2.resize(face_img, (112, 112))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img, (1, 112, 112, 1))
        result = model.predict(reshaped_img)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x+w, y+h), rect_color_dict[label], 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), rect_color_dict[label], -1)
        cv2.putText(img, text_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        if (label == 1):
            messagebox.showwarning('Access Denied! Please Wear a Face Mask')
            # message = f'Subject: {Subject} \n\n {Text}'
            # mail = smtplib.SMTP('smtp.gmail.com',587)
            # mail.ehlo()
            # mail.starttls()
            # mail.login('email@gmail.com', 'password')
            # mail.sendmail('email@gmail.com', 'receiver email',  message)
            # mail.close()
        else:
            pass
        break

    cv2.imshow('LIVE Video Feed', img)
    key = cv2.waitKey(1)
    if(key == 27):
        break
