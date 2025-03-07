import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

#
#model = YOLO("./yolo11s-finetuned-640px-by-2.pt")
# 
PATCH_SIZE = 640
OVERLAP = 10

detection_model_seg = AutoDetectionModel.from_pretrained(
    model_type='yolo11', # or 'yolov8'
    model_path='./yolo11s-finetuned.pt',
    confidence_threshold=0.5,
    device="cpu", # or 'cuda:0'
)

im = read_image("./data/mavic3m/campo1/DJI_202411071020_004_Crear-ruta-de-zona2/DJI_20241107102712_0085_D.JPG")
h = im.shape[0]
w = im.shape[1]

result = get_prediction("./data/mavic3m/campo1/DJI_202411071020_004_Crear-ruta-de-zona2/DJI_20241107102712_0085_D.JPG", detection_model_seg, full_shape=(h, w))

result.export_visuals(export_dir="./results/")

result = get_sliced_prediction(
    "./data/mavic3m/campo1/DJI_202411071020_004_Crear-ruta-de-zona2/DJI_20241107102712_0085_D.JPG",
    detection_model_seg,
    slice_height = 640,
    slice_width = 640,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

result.export_visuals(export_dir="./results/", file_name="pred2")


Image("./results/prediction_visual.png")