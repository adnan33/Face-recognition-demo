from keras_facenet import FaceNet
import pandas as pd
import numpy as np
import face_recognition
from annoy import AnnoyIndex
import os,glob,cv2,time,dlib,mtcnn



# defining the embedder 
embedder = FaceNet()
#loading the saved face name and features
face_df = pd.read_csv("Face_feature_database.csv")
face_names = face_df["Name"].values
face_feats = face_df["Face_features"].to_numpy()
face_feats_np = [np.fromstring(x[1:-1],sep=',') for x in face_feats]
#loading the annoy face database
face_db = AnnoyIndex(512,"dot")
face_db.load("face_db.ann")
face_width = 150
face_height = 150
THRES = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 1

def draw_rect_on_face(frame, name, face_location):
    top_left = (face_location[3], face_location[0])
    bottom_right = (face_location[1], face_location[2])

    # Get color by name using our fancy function
    color = name_to_color(name)
    # Paint frame
    cv2.rectangle(frame, top_left, bottom_right, color, FRAME_THICKNESS)
    # Now we need smaller, filled grame below for a name
    # This time we use bottom in both corners - to start from bottom and move 50 pixels down
    top_left = (face_location[3], face_location[2])
    bottom_right = (face_location[1], face_location[2] + 22)

    # Paint frame
    cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)

    # Wite a name
    cv2.putText(frame, name+"!!!!", (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
    
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

def detect_faces(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="hog")

def detect_faces_dlib(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    rects_conv = []
    for rect in rects:
        # Print the location of each face in this image
        rects_conv.append([rect.top(),rect.right(), rect.bottom(), rect.left()])
    return rects_conv
def detect_faces_mtcnn(frame):
    rects = detector2.detect_faces(frame)
    rects_conv = []
    for rect in rects:
        # Print the location of each face in this image
        rects_conv.append([rect['box'][1],rect['box'][0]+rect['box'][2],rect['box'][1]+rect['box'][3],rect['box'][0]])
    return rects_conv

def recognize_faces(frame,face_locations):
    for i in range(len(face_locations)):
        #top,right,bottom,left = face_locations[i]
        face = frame[face_locations[i][0]:face_locations[i][2], face_locations[i][3]:face_locations[i][1]]
        face = cv2.resize(face,(face_width,face_height))
        face = face.reshape(1,face_width,face_height,3)
        face_embedding = embedder.embeddings(face)
        sim = face_db.get_nns_by_vector(face_embedding[0],n=1,include_distances=True)
        if sim[1][0]>THRES:
            p_name = face_names[sim[0][0]]
        else:
            p_name = "Unknown"
        draw_rect_on_face(frame, p_name, face_locations[i])
    return frame

detector = dlib.get_frontal_face_detector()
detector2 = mtcnn.MTCNN()
video_capture = cv2.VideoCapture(0) 
while True:
    ret,frame = video_capture.read()
    print(frame.shape)
    f_time_start = time.time()
    face_locations = detect_faces_mtcnn(frame)
    f_time_end = time.time()
    if len(face_locations) > 0:
        frame = recognize_faces(frame,face_locations)
    r_time_end = time.time()
    cv2.imshow("Face Recognition Demo",frame)
    print(f"Face locating time={f_time_end-f_time_start}\n Face recognition time={r_time_end-f_time_end}")
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
    
    