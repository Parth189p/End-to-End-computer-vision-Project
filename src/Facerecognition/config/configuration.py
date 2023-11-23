import os
from pathlib import Path
from Facerecognition.constants import *
from Facerecognition.utils.common import read_yaml, create_directories
from Facerecognition.entity.config_entity import DataIngestionConfig ,PrepareBaseModelConfig,DataprocessConfig,PrepareCallbacksConfig,TrainingConfig,EvaluationConfig




class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_include_top=self.params.INCLUDE_TOP,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    
    def get_data_processing_config(self) -> DataprocessConfig:
        config = self.config.data_preprocessing
        data = os.path.join(self.config.data_ingestion.unzip_dir, "Facerecognition")
        create_directories([Path(config.root_dir)])
        
        data_preprocess_config = DataprocessConfig(
            root_dir=Path(config.root_dir),
            preprocessed_dir=config.preprocessed_dir,
            data = Path(data)
        )

        return data_preprocess_config
    
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config
    
       
    
    def get_training_config(self, data_preprocess_config: DataprocessConfig) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        training_data = data_preprocess_config.root_dir / "preprocessed_images"

        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=self.params.EPOCHES,
            params_batch_size=self.params.BATCH_SIZE,
            params_image_size=self.params.IMAGE_SIZE
        )

        return training_config 
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_preprocessing/preprocessed_images",
            all_params=self.params,
            mlflow_uri ="https://dagshub.com/Parth-cilans/Face-reg.mlflow",
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config