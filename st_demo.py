import streamlit as st
import numpy as np
import os, urllib, cv2
from keras_facenet import FaceNet
import pandas as pd
import numpy as np
import face_recognition
from annoy import AnnoyIndex
import os,glob,cv2,time,dlib,mtcnn



# defining the embedder 
embedder = FaceNet()
#loading the saved face name and features
df_filename = "Face_feature_database.csv"
idx_filename = "face_db.ann"
idx_shape = 512
face_width = 150
face_height = 150
THRES = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 1

def load_facedb():
    global face_df
    global face_feats
    global face_names
    global face_db
    face_df = pd.read_csv(df_filename)
    face_names = face_df["Name"].values
    face_feats = face_df["Face_features"].to_numpy()
    face_feats_np = [np.fromstring(x[1:-1],sep=',') for x in face_feats]
    #loading the annoy face database
    face_db = AnnoyIndex(idx_shape,"dot")
    for i,face in enumerate(face_feats_np):
    	face_db.add_item(i,face)
    face_db.build(10)
  

def insert_new_face(image,name):
    global face_df
    global face_db
    face_db.unload()
    if (name not in face_names):
        face_locations = detect_faces(image)
        face = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
        face = cv2.resize(face,(224,224))
        face = face.reshape(1,224,224,3)
        face_feat = embedder.embeddings(face)[0]
        face_feats_np = [np.fromstring(x[1:-1],sep=',') for x in face_feats]
        encoded_face_db = dict(zip(face_names,face_feats_np))
        encoded_face_db[name] = face_feat
        face_feats_np.append(face_feat)
        #print(os.access(idx_filename, os.W_OK),os.access(idx_filename, os.R_OK))
        face_db = AnnoyIndex(idx_shape,"dot")
        #print(os.access(idx_filename, os.W_OK),os.access(idx_filename, os.R_OK))
        for i,face in enumerate(face_feats_np):
            face_db.add_item(i,face)
        
        #print(os.access(idx_filename, os.W_OK),os.access(idx_filename, os.R_OK))
        #face_db.on_disk_build(idx_filename)
        face_db.build(10)
        
        face_df = pd.DataFrame()
        face_df["Name"] = encoded_face_db.keys()
        face_df["Face_features"] = encoded_face_db.values()
        face_df.to_csv(df_filename ,index = False)
        
        del face_df
        del face_db
        load_facedb()
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
def show_db():
    st.subheader('List of Known People')
    face_df_show = pd.DataFrame(face_names, columns=["Known People"])
    st.table(face_df_show)

def show_live_stream():
    st.subheader('Video Stream')
    video_capture = cv2.VideoCapture(0) 
    imbox = st.empty()
    while True:
        ret,frame = video_capture.read()
        face_locations = detect_faces(frame)
        if len(face_locations) > 0:
            frame = recognize_faces(frame,face_locations)
        imbox.image(frame, channels = "BGR")
def add_face():
    st.subheader('Register New Person')
    name = st.text_input("Name")
    uploaded_file = st.file_uploader("Image", type=['jpg','png'])
    print(uploaded_file)
    if uploaded_file!= None and uploaded_file!=[] and name!='' :
        print(name)
        up_image = uploaded_file.read()
        nparr = np.frombuffer(up_image, np.uint8)
        up_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        insert_new_face(up_image,name)
        del name
        del up_image
        st.error("Face Added")
        if st.button("Redirect To Known People"):
            sidebar_option=app_options[1]
    

st.title("Face Recognition demo")

st.sidebar.title("SideBar")
app_options = ["Live Video","Known People","Add New Face","Remove Known Face","Show Log"]
sidebar_option = st.sidebar.selectbox("Options",options=app_options)

load_facedb()
detector = dlib.get_frontal_face_detector()


if sidebar_option==app_options[0]:
    show_live_stream()
elif sidebar_option==app_options[1]:
    show_db()
elif sidebar_option==app_options[2]:
    add_face()