
#for model deployement 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from keras.optimizers import Adam #important for loading model
import cv2
import numpy as np
import face_recognition
import argparse
import json

AUTOTUNE = tf.data.AUTOTUNE

def infer(original_image):
    image = cv2.resize(original_image, dsize=(600, 400))
    image = img_to_array(image)
    image = image[:, :, :3] if image.shape[-1] > 3 else image
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = np.array(output_image) 
    enhanced_image = output_image[:, :, ::-1].copy() 
    return enhanced_image

def pil_to_cv2(pil_img_object):
    cv2_img = np.array(pil_img_object) 
    cv2_img = cv2_img[:, :, ::-1].copy() 
    return cv2_img

zero_dce_model = tf.keras.models.load_model('lol_model',compile=False)
def faceRecognition(source='0',
                    input_file_path='0', 
                    output_folder_path="./"):
    
    classNames = []
    encodeListKnown = []
    for encoding in os.listdir("./encodings"):
        classNames.append(encoding.split(".")[0])
        encode = np.load("./encodings/"+encoding)
        encodeListKnown.append(encode)
    
    if source=='0':
        print("opening webcam")
        cap = cv2.VideoCapture(0)
        
        while True:
            success, img = cap.read()
            img = cv2.resize(img, dsize=(600, 400))
            img1 = infer(img)
            imgS = cv2.resize(img1,(0,0),None,0.25,0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
            print("Press B on webcam to exit")
            for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                matchIndex = np.argmin(faceDis)
            
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
            
                cv2.imshow('webcam',img)
            if cv2.waitKey(10) == ord('b'):
                print("webcam capturing done")
                break
        cap.release()
        cv2.destroyAllWindows()
    elif source=='1':
        img = cv2.imread(input_file_path)
        img = cv2.resize(img, dsize=(600, 400))
        img1 = infer(img.copy())
        imgS = cv2.resize(img1,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        if len(facesCurFrame)==0:
            imgS = cv2.resize(img,(0,0),None,0.25,0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
        print(facesCurFrame)
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):

            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            matchIndex = np.argmin(faceDis)
            print(matches)
            print(matchIndex)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)
                img_name= input_file_path.split("/")[-1]
                print(output_folder_path+"/"+img_name)
                cv2.imwrite(output_folder_path+"/"+img_name,img)
                print("image saved in the directory: "+ output_folder_path)
        else:
            print("process done")  
    else:
        print("invalid input")     

def main(source='0',
         input_file_path='0', 
         output_folder_path="./face_detected"):
    
    faceRecognition(source,
                    input_file_path, 
                    output_folder_path)
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="can be 0/1, 0 for webcam, 1 for image")
    parser.add_argument("-input_file_path", help="Applicable if source 1,Image path of the person with face.")
    parser.add_argument("-output_folder_path", help="Applicable if source 1, Name of the person")
    args = parser.parse_args()
    input_file_path = ""
    output_folder_path="./face_detected"
    if not args.source=='0':
        input_file_path = args.input_file_path 
        if not args.output_folder_path is None:
            output_folder_path = args.output_folder_path
            if not os.path.isdir(output_folder_path):
                os.makedirs(output_folder_path)
                print("made directory")

    main(args.source,input_file_path,output_folder_path)

    # main()
