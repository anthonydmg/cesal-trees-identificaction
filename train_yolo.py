from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s-seg.pt")

    # Entrenar el modelo
    model.train(data="./avocado_data_640px_by_2.yaml", epochs=40, imgsz=640, batch = 16)

    model.val()