import urllib.request as request
import zipfile
from Facerecognition import logger
from Facerecognition.utils.common import  get_size
import cv2
import os
import numpy as np
from PIL import Image
from Facerecognition.entity.config_entity import DataprocessConfig



class DataPreprocess:
    def __init__(self, config: DataprocessConfig):
        self.config = config

    def preprocess_data(self, data_preprocess_config):
        image_width = 224
        image_height = 224
        cascade_file_path = os.path.abspath('haarcascade_frontalface_default.xml')
        facecascade = cv2.CascadeClassifier(cascade_file_path)
        images_dir = data_preprocess_config.data

        # Create a new directory for preprocessed images within the root directory
        preprocessed_images_dir = data_preprocess_config.root_dir / "preprocessed_images"
        preprocessed_images_dir.mkdir(parents=True, exist_ok=True)

        current_id = 0
        label_ids = {}

        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.endswith(("png", "jpg", "jpeg")):
                    # Path of the image
                    path = os.path.join(root, file)

                    # Get the label name (name of the person)
                    label = os.path.basename(root).replace(" ", ".").lower()

                    # Create a subdirectory for each class (label)
                    class_dir = preprocessed_images_dir / label
                    class_dir.mkdir(parents=True, exist_ok=True)

                    # Load the image
                    imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
                    image_array = np.array(imgtest, "uint8")

                    # Get the faces detected in the image
                    faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5)

                    # If not exactly 1 face is detected, skip this photo
                    if len(faces) != 1:
                        print(f'---Photo skipped---\n')
                        continue

                    # Save the detected face(s) in the class subdirectory
                    for (x_, y_, w, h) in faces:
                        # Resize the detected face to 224x224
                        size = (image_width, image_height)

                        # Detected face region
                        roi = image_array[y_: y_ + h, x_: x_ + w]

                        # Resize the detected head to the target size
                        resized_image = cv2.resize(roi, size)

                        preprocessed_image_path = class_dir / f"{current_id}.jpg"
                        im = Image.fromarray(resized_image)
                        im.save(preprocessed_image_path)
                        current_id += 1
