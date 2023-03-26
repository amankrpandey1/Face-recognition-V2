import os
import cv2
import argparse
from json import JSONEncoder
import numpy
from face_recognition import face_locations,face_encodings

def generate_encoding(img):
    facesCurFrame = face_locations(img)
    if len(facesCurFrame) !=0:

        print("Face Found in the image. Registering the face")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_encodings(img)[0]

        return True, encode
    else:
        return False, []

def register_face(input_file_path = "",
                  classname = ""):
        
    for img,name in zip(input_file_path.split(","),classname.split(",")):
        try:
            img = cv2.imread(input_file_path)
            img = cv2.resize(img, dsize=(600, 400))
        except: 
            print("Invalid image for "+input_file_path+". Try Again!")
            continue
        result, encoding = generate_encoding(img)
        if result:
            numpy.save("./encodings/"+classname,encoding)
            cv2.imwrite("./ImageFace/"+name+".jpg",img)

        else:
            print("Face not found in the image "+ input_file_path+ ". Try Again with new image!")
            pass
       
        print("entry done")
        return "True"
    
    
# register_face("./Aman_Pandey.png","Aman")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path", help="Image path of the person(s) with visible face, can be './a.png,./b.png'")
    parser.add_argument("classname", help="Name of the person(s), can be 'aman,chirag'")

    args = parser.parse_args()    
    

    register_face(input_file_path=args.input_file_path,
                  classname=args.classname)
