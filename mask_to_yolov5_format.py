from glob import glob
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
def mask_to_yolo(mask, save_path, class_id):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    with open(save_path, "w") as f:
        for contour in contours:
            if len(contour) <= 10: # Poligono invalido
                continue
            ## Normalizar coordenadas
            # Simplificar el contorno
            epsilon = 0.004 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Reducir si hay mas de 32 puntos
            max_points = 64
            if len(approx) > max_points:
                step = len(approx) // max_points
                approx = approx[::step][:max_points]
            # Asegurar la forma correcta (N,2)
            approx = np.squeeze(approx)
            if len(approx) <= 10: # Poligono invalido
                continue
            # Eliminar dimensión innecesaria pero asegurando que sea (N, 2)
            #contour = np.squeeze(contour)
            if approx.ndim == 1:  # Si solo hay un punto, lo convertimos a (1,2)
                approx = np.expand_dims(approx, axis=0)

            normalized_points = [(x/w, y/h) for x, y in approx]
            flattened_points = [str(coord) for point in normalized_points for coord in point]
            # Escribit en formato yolo
            f.write(f"{class_id} " + " ".join(flattened_points) + "\n")


def gen_slicing_images_yolo_format(im_paths, save_dir, split_name = "train", slice_w = 640, slice_h = 640):
    for im_path in tqdm(im_paths, desc = f"{split_name.upper()} Images"):
        im = cv2.imread(im_path)
        base_name_file = os.path.basename(im_path)[:-4]
        mask = cv2.imread(f"{dir_base}/masks/{base_name_file}_MASK.JPG")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask != 0
        mask = mask.astype(np.uint8) * 255

        ## Redimensionar images y mascara
        im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2), interpolation=cv2.INTER_LANCZOS4)
        #print(mask.shape)
        #cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Permite redimensionar la ventana
        #cv2.imshow("Image", mask)
        #cv2.resizeWindow("Image", 600, 400)  # Ancho y alto deseados
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        height, width = im.shape[:2]
        n_h = height // slice_h + 1
        n_w = width // slice_w + 1
        im_padded = np.zeros((n_h * slice_h, n_w * slice_w, 3), dtype= np.uint8)
        im_padded[:height,:width,:] = im

        mask_padded = np.zeros((n_h * slice_h, n_w * slice_w), dtype= np.uint8)
        mask_padded[:height,:width] = mask

        im_slices = [im_padded[j*slice_h:(j+1)*slice_h, i*slice_w:(i+1)*slice_w,:].copy() for i in range(n_w) for j in range(n_h)]
        mask_slices = [mask_padded[j*slice_h:(j+1)*slice_h, i*slice_w:(i+1)*slice_w].copy() for i in range(n_w) for j in range(n_h)]

        for id, im_s in enumerate(im_slices):
            cv2.imwrite(f"{save_dir}/images/{split_name}/{base_name_file}_SLICE_{id}.jpg", im_s)
            cv2.imwrite(f"{save_dir}/masks/{split_name}/{base_name_file}_SLICE_{id}.jpg", mask_slices[id])
            mask_to_yolo(mask_slices[id],f"{save_dir}/labels/{split_name}/{base_name_file}_SLICE_{id}.txt", 0)

dir_base = "./data/ds-avocado"

images_paths = glob(f"{dir_base}/images/*.JPG")

train_im_paths, val_im_paths = train_test_split(images_paths, test_size=0.25, random_state=42, shuffle=True)
SLICE_SIZE = 640

save_dir = f"./datasets/avocado-yolo-format-{SLICE_SIZE}-by-2"

os.makedirs(f"{save_dir}/masks/train", exist_ok=True)
os.makedirs(f"{save_dir}/masks/val", exist_ok=True)
os.makedirs(f"{save_dir}/images/train", exist_ok=True)
os.makedirs(f"{save_dir}/images/val", exist_ok=True)
os.makedirs(f"{save_dir}/labels/train", exist_ok= True)
os.makedirs(f"{save_dir}/labels/val", exist_ok= True)

gen_slicing_images_yolo_format(train_im_paths, save_dir, split_name = "train", slice_w = SLICE_SIZE, slice_h = SLICE_SIZE)
gen_slicing_images_yolo_format(val_im_paths, save_dir, split_name = "val", slice_w = SLICE_SIZE, slice_h = SLICE_SIZE)