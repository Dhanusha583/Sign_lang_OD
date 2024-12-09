import os
import sys
import yaml
import zipfile
import shutil
from signLanguage.utils.main_utils import read_yaml_file
from signLanguage.logger import logging
from signLanguage.exception import SignException
from signLanguage.entity.config_entity import ModelTrainerConfig
from signLanguage.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Define the zip file path and extraction directory
            zip_file_path = r"D:\Projects\Sing_lang_OD\Sing_lang_OD\artifacts\12_09_2024_15_02_40\data_ingestion\data1_zip.zip"
            extraction_dir = r"D:\Projects\Sing_lang_OD\Sing_lang_OD"

            # Unzipping the data
            logging.info("Unzipping data...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_dir)
            os.remove(zip_file_path)  # Remove the zip file after extraction

            # Verify `data.yaml` exists in the extracted directory
            data_yaml_path = os.path.join(extraction_dir, "data.yaml")
            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError(f"{data_yaml_path} does not exist.")

            with open(data_yaml_path, 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            # Read and update the model configuration
            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            logging.info(f"Model config file name: {model_config_file_name}")
            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)
            custom_config_path = f'yolov5/models/custom_{model_config_file_name}.yaml'

            with open(custom_config_path, 'w') as f:
                yaml.dump(config, f)

            # Run YOLOv5 training
            logging.info("Starting YOLOv5 training...")
            os.system(
                f"cd yolov5 && python train.py --img 640 --batch {self.model_trainer_config.batch_size} "
                f"--epochs {self.model_trainer_config.no_epochs} --data ../data.yaml "
                f"--cfg ./models/custom_yolov5s.yaml --weights {self.model_trainer_config.weight_name} "
                f"--name yolov5s_results --cache"
            )

            # Copy the best model to the desired location
            best_model_src = "yolov5/runs/train/yolov5s_results/weights/best.pt"
            best_model_dest = os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy(best_model_src, best_model_dest)
            shutil.copy(best_model_src, "yolov5/")

            # Cleanup unnecessary files and folders
            shutil.rmtree("yolov5/runs", ignore_errors=True)
            if os.path.exists(data_yaml_path):
                os.remove(data_yaml_path)

            # Create ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise SignException(e, sys)
