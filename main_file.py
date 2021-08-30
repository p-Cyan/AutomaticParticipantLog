import datetime
import time

import cv2
import numpy as np
from keras import models
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC

# path for pretrained resnet ssd model
# citation, https://github.com/LZQthePlane/Face-detection-base-on-ResnetSSD
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"

resnet = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# loading pretrained facenet model
# citation, https://github.com/nyoki-mtl/keras-facenet
facenet = models.load_model('models/facenet_keras.h5')


# Capture video from file
cap = cv2.VideoCapture(0)

# load train dataset embeddings and labels
image_embeddings = np.load("save_files/embeddings.npy")
labels = np.load("save_files/labels.npy")

# normalize embeddings with l2 norm
embedding_encoder = Normalizer(norm='l2')
image_embeddings = embedding_encoder.transform(image_embeddings)

# encode labels to class numbers
label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels = label_encoder.transform(labels)

# use Support Vector Classifier to classify these embeddings
SVC_classifier = SVC(kernel='linear', probability=True)
SVC_classifier.fit(image_embeddings, labels)

# statistics for Logging data

# counts number of faces in each frame
face_count = 0

# set of people when last checked
previous_set_of_people = set()

# set of people when currently checked
current_set_of_people = set()

current_time = time.time()


def find_embeddings(facenet, face):
    """
    converts extracted image into face embeddings
    :param facenet: facenet model
    :param face: image of face
    :return:
    """

    # standardize and scale images so the embeddings wont be abnormal
    face = face.astype('float32')
    mean = face.mean()
    std = face.std()
    face = (face - mean) / std

    # model takes 1 extra dimension so expand dimensions
    face = np.expand_dims(face, axis=0)

    # predict the 128 embedding
    prediction = facenet.predict(face)
    return prediction[0]


def classify_image(facenet, image):
    """
    classifies given image by finding its embedding and then predicting its class based on SVC
    :param facenet: facenet model
    :param image: face image from camera feed
    :return: possible name of the person in image
    """

    # find embeddings of image
    new_embeddings = find_embeddings(facenet, image)

    # classify and label
    prediction = SVC_classifier.predict_proba([new_embeddings])
    label = max(prediction[0])
    name = ["unknown"]
    # if probability of all labels < 80% then lets consider the person unknown
    if label > 0.8:
        name = np.argmax(prediction[0])
        name = label_encoder.inverse_transform([name])
        return name
    return name


def add_departured_entry_to_log(set_of_entries):
    """
    add ndata to log about candidates left from screen
    :param set_of_entries: departured candidates list
    :return: None
    """
    f = open("LogBook.csv", "a")
    for i in set_of_entries:
        if i != "unknown":
            f.write(i + ",left_at," + str(datetime.datetime.now()) + "\n")
    f.close()


def add_arrival_entry_to_log(set_of_entries):
    """
    add ndata to log about candidates who newly arrived on screen
    :param set_of_entries: departured candidates list
    :return: None
    """
    f = open("LogBook.csv", "a")
    for i in set_of_entries:
        if i != "unknown":
            f.write(i + ",arrived_at," + str(datetime.datetime.now()) + "\n")
    f.close()


def no_of_faces(faces):
    """
    counts no.of faces currently on screen
    :param faces: current frame data
    :return: count of no of faces on screen
    """
    count = 0
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.7:
            count += 1
    return count


def reset_set(previous_set_of_people, current_set_of_people, face_count):
    """
    checks previous set and current set, if any person is changes, add the change to logbook
    :param previous_set_of_people: people found on last check
    :param current_set_of_people: people found on current check
    :param face_count: no.of people on screen
    :return: reset all values
    """
    # add departured entries to logbook
    departured_entries = previous_set_of_people - current_set_of_people
    add_departured_entry_to_log(departured_entries)

    # add arrived entries to logbook
    new_entries = current_set_of_people - previous_set_of_people
    add_arrival_entry_to_log(new_entries)
    previous_set_of_people = current_set_of_people
    return previous_set_of_people, set(), 0

# continuously check camera feed
while True:
    # read from webcam
    ret, frame = cap.read()
    h1, w1 = frame.shape[:2]

    # resize frame and turn the frame into blob
    frame = cv2.resize(frame, (300, 300))
    h2, w2 = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 117.0, 123.0))

    # send blob into resnet
    resnet.setInput(blob)
    faces = resnet.forward()

    # count no.of faces
    new_face_count = no_of_faces(faces)

    # if the no.of people in screen changed from previous check , find who changed
    if new_face_count != face_count:
        face_count = new_face_count

        # iterate through all outputs
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]

            # if confidence of the image being face is high, check identity
            if confidence > 0.7:
                box = faces[0, 0, i, 3:7] * np.array([w2, h2, w2, h2])
                (x, y, x1, y1) = box.astype("int")
                name = classify_image(facenet, cv2.resize(frame[y:y1, x:x1], (160, 160)))
                current_set_of_people.add(name[0])
                # cv2.putText(frame,name[0],(x-2,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                # cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

    # every 10 seconds, see who changed from the screen and add them to log
    if time.time() - current_time > 10:
        previous_set_of_people, current_set_of_people, face_count = reset_set(previous_set_of_people,
                                                                              current_set_of_people, face_count)
        current_time = time.time()
    # frame = cv2.resize(frame, (w1,h1))
    if ret == True:
        # cv2.imshow('frame',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
