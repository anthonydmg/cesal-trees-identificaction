import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
#
model = YOLO("./yolo11s-finetuned-640px-by-2.pt")
# 
PATCH_SIZE = 640
OVERLAP = 10

def split_image_into_patches(image, patch_size = PATCH_SIZE, overlap = OVERLAP):
    h, w, _ = image.shape
    step = patch_size - overlap

    patches = []
    coordinates = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = image[y:y + patch_size, x:x+patch_size]
            # Si el parce es menor que el tamnaio esperado, rellenar
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.copyMakeBorder(patch, 0, patch_size - patch.shape[0], 0, patch_size - patch.shape[1], cv2.BORDER_CONSTANT, value=(0,0,0))

            patches.append(patch)
            coordinates.append((x,y))
    
    return patches, coordinates, (w, h)

def visualize_patch_results(patch, result):
    patch_viz = patch.copy()

    # Dibujar bounding boxes y máscaras
    print("result.masks:", result.masks)
    if result.masks is not None:
        for mask in result.masks.xy:
            polygon = np.array(mask, dtype=np.int32)
            cv2.polylines(patch_viz, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    
    if result.boxes is not None:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(patch_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar el parche con detecciones
    cv2.imshow("Detecciones en el parche", patch_viz)
    cv2.waitKey(0)
    
def run_inference_on_patches(patches):
    results = []
    for patch in patches:
        #patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        result =  model(patch, imgsz = PATCH_SIZE, conf = 0.4, verbose=False)
        # Visualizar resultados en el parche
        #visualize_patch_results(patch, result[0])
        results.append(result[0])

    return results

def merge_predictions(results, coordinates, image_size, output_dir = None):
    w, h = image_size
    final_mask = np.zeros((h,w), dtype=np.uint8)
    i = 0
    if output_dir:
        os.makedirs(f"{output_dir}/patches", exist_ok=True)
    for result, (x_offset, y_offset) in zip(results, coordinates):
        i+=1
        if output_dir:
            result.save(filename=f"{output_dir}/patches/patch_{i}_x{x_offset}_y{y_offset}.jpg")  # save to disks
        if result.masks is None:
            continue

        for mask in result.masks.xy:
            polygon = np.array(mask, dtype=np.int32)
            polygon[:, 0] += x_offset
            polygon[:, 1] += y_offset
            # Dibujar mascara
            cv2.fillPoly(final_mask, [polygon], color=255)
    
    return final_mask


def segment_large_image(image_path, output_dir = None):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_LANCZOS4)
    # Dividir la imagen en parches
    patches, coordinates, image_size = split_image_into_patches(image, patch_size=PATCH_SIZE, overlap=OVERLAP)
    
    # Inferencia en cada parche
    results = run_inference_on_patches(patches)
    # Fusionar predicciones
    final_mask = merge_predictions(results, coordinates, image_size, output_dir)
    # Guardar la mascara resultante
    if output_dir:
        cv2.imwrite(f"{output_dir}/output_mask.png", final_mask)
        print(f"Máscara de segmentación guardada en {output_dir}")

segment_large_image(image_path="./data/mavic3m/campo1/DJI_202411071020_004_Crear-ruta-de-zona2/DJI_20241107102712_0085_D.JPG",
                    output_dir="./prediction-by-2")