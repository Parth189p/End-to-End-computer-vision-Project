from Facerecognition.config.configuration import ConfigurationManager
from Facerecognition.components.preprocess_data import DataPreprocess
from Facerecognition import logger


STAGE_NAME = "Preprocess Data"

class DataPreprocessingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_preprocess_config = config.get_data_processing_config()
        data_preprocess = DataPreprocess(config=data_preprocess_config)
        data_preprocess.preprocess_data(data_preprocess_config)
        
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e