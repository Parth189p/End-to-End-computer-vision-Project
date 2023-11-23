import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import keras
from pathlib import Path
from keras import layers
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
from Facerecognition.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    
    def get_base_model(self):
        self.model = VGGFace(
            model='vgg16',
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
        
        
    @staticmethod
    def _prepare_full_model(model,freeze_all, freeze_till, classes,):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:freeze_till]:
                model.trainable = False
            for layer in model.layers[freeze_till:]:
                model.trainable = True
            
            
        x = model.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)

        # final layer with softmax activation
        prediction = Dense(
            units=classes, activation='softmax')(x)


        full_model = Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=None,
            freeze_till=19,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    