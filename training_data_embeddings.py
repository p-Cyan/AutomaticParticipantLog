import os

import cv2
import numpy as np
from keras import models

# path for our intial model data
rootdir = 'data/train'

# path for pretrained resnet ssd model
# citation, https://github.com/LZQthePlane/Face-detection-base-on-ResnetSSD
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
# loading resnet model into cv2
resnet = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# loading pretrained facenet model
# citation, https://github.com/nyoki-mtl/keras-facenet
facenet = models.load_model('models/facenet_keras.h5')


def detect_and_extract_faces(resnet, image):
    """
    uses resnet to extract and return faces from images
    :param resnet: resnet model
    :param image: the image from which we need to extract face
    :return: extracted face
    """

    # resnet only takes 300x300 image so we need to resize it
    image = cv2.resize(image, (300, 300))

    # convert into blob and apply mean subtraction
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 117.0, 123.0))

    # use resnet to find faces
    resnet.setInput(blob)
    faces = resnet.forward()
    h, w = image.shape[:2]

    # iterate over all outputs
    for i in range(faces.shape[2]):

        # check confidence if its a face. if > 0.5 consider it as really a face.
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            # find the bounding box
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])

            # extract bounding box pixels from image
            (x, y, x1, y1) = box.astype("int")

            # resize to 160 x 160 as facenet model requires those dimensions
            return cv2.resize(image[y:y1, x:x1], (160, 160))


def find_data_in_path(resnet, file_path):
    """
    Loads images from given path and returns their embedding and labels
    :param net: resnet model
    :param file_path: path for image directories
    :return: image embeddings and labels
    """

    # lists for storing all image and label embeddings
    images = []
    labels = []

    # for each file in each directory
    for subdir, dirs, files in os.walk(file_path):
        for file in files:
            path = subdir.split("\\")
            filename = file
            fullpath = path[0] + "/" + path[1] + "/" + filename

            # read image
            im = cv2.imread(fullpath)

            # extract facenet embeddings
            im1 = detect_and_extract_faces(resnet, im)

            # add image and labels to list
            images.append(im1)
            labels.append(path[1])

    # convert to numpy arrays
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels


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


# extract images and labels from file path
images, labels = find_data_in_path(resnet, rootdir)

# find embeddings for each image
image_embeddings = []
for i in images:
    image_embeddings.append(find_embeddings(facenet, i))
# convert to numpy array
image_embeddings = np.asarray(image_embeddings)

# save embeddings for later use
np.save("save_files/embeddings", image_embeddings)
np.save("save_files/labels", labels)
