from ..common.generator import Generator
from ..common.dataset import VQADataset
from ..common.utils import Logger
from .report import VQAReporter
from tqdm import tqdm
import os

# Important Data for Linux (Colab)
name = "val"
annotationsJSON = "annotations/filtered_answers.json"
questionsJSON = "questions/filtered_questions.json"
imageDirectory = "<INSERT CLEAN IMAGE DIRECTORY>"
imagePrefix = None
outputPath = "<INSERT OUTPUT DIRECTORY>"
logPath = "Data_Mixed_Noise/"
# reportPath = "Data/Reports/"


# Creating a logger
logger = Logger(logPath)
logger.info("Starting experiment.")


# Transformation of dataset
dataset = VQADataset(name, questionsJSON, annotationsJSON, imageDirectory, imagePrefix, logger)
logger.info("VQA2.0 dataset loaded.")

generator = Generator(dataset, logger)
transformationsList = ["Contrast_L1","Contrast_L2","Contrast_L3","Contrast_L4","Contrast_L5"]
# transformationsList = list(generator.validTransformations.keys())[23:]
generator.transform(transformationsList, outputPath=outputPath)
