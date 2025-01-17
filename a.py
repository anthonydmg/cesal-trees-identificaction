import os
import cv2
import numpy as np
from glob import glob
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from PIL import Image
cv2.ocl.setUseOpenCL(False)

folder_path = "./data/trees-avocado/m3m/campo2/images/"

jpg_files = glob(f"{folder_path}/*.JPG")

images = jpg_files[:2]
# 📌 2. Usar OpenCV para ensamblar las imágenes en un ortomosaico
stitcher = cv2.Stitcher_create()
status, ortomosaico = stitcher.stitch([cv2.imread(img) for img in images])
plt.imshow(cv2.cvtColor(ortomosaico, cv2.COLOR_BGR2RGB))
plt.title(f"Mosaico progresivo después de la imagen {i+2}")
plt.show()

if status == cv2.Stitcher_OK:
    cv2.imwrite("ortomosaico.jpg", ortomosaico)
    print("Ortomosaico guardado correctamente.")
