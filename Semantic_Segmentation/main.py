from src.pipeline.stage_04_data_loader import DataLoaderPipeline
from src.pipeline.stage_05_model import ModelPipeline
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.pipeline.stage_03_dataset import DatasetPipeline

if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        
        # --- Data Ingestion Stage ---
        STAGE_NAME = "Data Ingestion Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline(config=config_manager)
        data_ingestion_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Data Transformation Stage ---
        STAGE_NAME = "Data Transformation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation_pipeline = DataTransformationPipeline(config=config_manager)
        transform = data_transformation_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  

        # --- Dataset Preparation Stage ---
        STAGE_NAME = "Dataset Preparation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        dataset_pipeline = DatasetPipeline(config=config_manager, transform=transform)
        train_dataset, val_dataset = dataset_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Data Loader Stage ---
        STAGE_NAME = "Data Loader Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_loader_pipeline = DataLoaderPipeline(config=config_manager, train_dataset=train_dataset, val_dataset=val_dataset)
        train_loader, val_loader = data_loader_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Build Model Stage ---
        STAGE_NAME = "Build Model Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_pipeline = ModelPipeline(config=config_manager)
        model = model_pipeline.build_model()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Train Model Stage ---
        STAGE_NAME = "Train Model Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_pipeline = ModelPipeline(config=config_manager)
        model_pipeline.run_pipeline(model=model, train_loader=train_loader, val_loader=val_loader)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.error(f"Error occurred during {STAGE_NAME} stage: {e}")
        raise e
