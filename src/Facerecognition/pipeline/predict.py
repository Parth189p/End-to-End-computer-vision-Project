import cv2
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import imagenet_utils as utils

class PredictionPipeline:
    def __init__(self, filenames):
        self.filenames = filenames
        cascade_path = os.path.join(r"haarcascade_frontalface_default.xml")
        self.model = load_model(os.path.join("artifacts" ,"training", "model.h5"))
        self.facecascade = cv2.CascadeClassifier(cascade_path)

    def predict(self):
        predictions = []

        for file_reference in self.filenames:
            imgtest = cv2.imread(file_reference, cv2.IMREAD_COLOR)
            if imgtest is None:
                print(f"Failed to read the image from {file_reference}")
                continue

            image_array = np.array(imgtest, "uint8")

            faces = self.facecascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:  # Check if faces is empty
                print(f'---No face detected in {file_reference}; photo skipped---')
                predictions.append({file_reference: "No Face Detected"})
                continue

            for (x_, y_, w, h) in faces:
                size = (224, 224)
                roi = image_array[y_: y_ + h, x_: x_ + w]
                resized_image = cv2.resize(roi, size)

                x = image.img_to_array(resized_image)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x)

                result = np.argmax(self.model.predict(x), axis=1)
                # print(result)

                if result[0] == 0:
                    prediction = 'Emma_Watson'
                elif result[0] == 1:
                    prediction = 'Narendra_Modi'
                elif result[0] == 2:
                    prediction = 'Tom_Cruise'
                else:
                    prediction = 'Virat_kohli'

                predictions.append({file_reference: prediction})

        return predictions

    
    

