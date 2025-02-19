from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s-seg.pt")

    # Entrenar el modelo
    model.train(data="./avocado_data.yaml", epochs=50, imgsz=640, batch = 16)

    model.val()