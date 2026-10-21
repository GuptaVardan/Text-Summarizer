"""Run ingestion, transformation, fine-tuning and held-out evaluation."""
from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_3_model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.textSummarizer.pipeline.stage_4_model_evaluation import ModelEvaluationTrainingPipeline


def main():
    stages = (
        ("Data ingestion", DataIngestionTrainingPipeline().initiate_data_ingestion),
        ("Data transformation", DataTransformationTrainingPipeline().initiate_data_transformation),
        ("Model training", ModelTrainerTrainingPipeline().initiate_model_trainer),
        ("Model evaluation", ModelEvaluationTrainingPipeline().initiate_model_evaluation),
    )
    for name, run_stage in stages:
        logger.info("Starting %s", name)
        try:
            run_stage()
        except Exception:
            logger.exception("%s failed", name)
            raise
        logger.info("Completed %s", name)


if __name__ == "__main__":
    main()
