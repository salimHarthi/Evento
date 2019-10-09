from pyagender import PyAgender
import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pandas as pd
import numpy as np
import datetime

agender = PyAgender()

df = pd.DataFrame(columns= ['name','age', 'gender','time_in','time_out'])

def predict( imgMe,path,knn_clf=None, model_path=None, distance_threshold=0.6):
 
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = imgMe
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
nameold=[]
while 1:
    
    ret,img = cap.read()
    if ret:
        faces = agender.detect_genders_ages(img)

        for face in faces:
            size=[face['width'],face['height'],face['top'],face['left']]
            #print(faces)
            gender=round(face['gender'])
            age=face['age']
            #print(gender)
            if gender>0:
                gender='female'
            else:
                gender='male'
            full_file_path=(size[3],size[2]),(size[3]+size[1],size[2]+size[0])
            predictions = predict(img,full_file_path, model_path="trained_knn_model.clf")
             # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                cv2.putText(img,name, (int(size[3]),int(size[2]-50)), cv2.FONT_HERSHEY_SIMPLEX , 0.66,(0,0,255), 2,cv2.LINE_AA)
                if not(name in nameold):
                    nameold.append(name)
                    now = datetime.datetime.now()
                    df = df.append({'name':name,'age': age, 'gender':gender,'time_in':now,'time_out':0}, ignore_index=True)
                    df1 = df.append({'age': age, 'gender':gender,'avg time':0}, ignore_index=True)
            
            
            print(df)
            cv2.rectangle(img,(size[3],size[2]),(size[3]+size[1],size[2]+size[0]),(0,0,255),2)
            cv2.putText(img,gender, (int(size[3]),int(size[2]-10)), cv2.FONT_HERSHEY_SIMPLEX , 0.66,(0,0,255), 2,cv2.LINE_AA) #writing the distenc on the img
            cv2.putText(img,str(age), (int(size[3]),int(size[2]-30)), cv2.FONT_HERSHEY_SIMPLEX , 0.66,(0,0,255), 2,cv2.LINE_AA) #writing the distenc on the img
            
            
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
df.to_csv('data.csv', sep='\t', encoding='utf-8')
df1.to_csv('data_for_sell.csv', sep='\t', encoding='utf-8')
