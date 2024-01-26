# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:23:06 2024

@author: DS.Salma
"""
import cv2

# تفعيل الكاميرا
cap = cv2.VideoCapture(0)

# استخدام مُصنف الوجوه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    # تحويل الإطار إلى درجات الرمادي لمعالجتها بشكل أفضل
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # البحث عن الوجوه في الإطار
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # رسم مستطيل حول كل وجه مكتشف
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
    
    # عرض الإطار المعالج
    cv2.imshow('Face Detection', frame)
    
    # الانتظار للضغط على مفتاح ESC لإنهاء البرنامج
    if cv2.waitKey(1) & 0xFF == 27:
        break

# إغلاق الكاميرا وتدمير النوافذ
cap.release()
cv2.destroyAllWindows()
