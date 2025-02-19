import matplotlib.pyplot as plt
import random

def plot_yolo_polygons(file_path, image_size=1000):
    """
    Lee un archivo de anotaciones YOLO con polígonos y los grafica con diferentes colores.
    
    Parámetros:
    - file_path: Ruta del archivo de texto con anotaciones YOLO.
    - image_size: Tamaño de la imagen en la que se escalarán las coordenadas (default: 1000x1000).
    """
    plt.figure(figsize=(8, 8))

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        values = list(map(float, line.split()))
        if len(values) < 3:
            continue  # Ignorar líneas inválidas

        class_id = values[0]  # La primera columna es la clase (no se usa aquí)
        coords = values[1:]   # Coordenadas x, y del polígono

        # Separar las coordenadas en listas de X e Y
        x_points = coords[0::2]
        y_points = coords[1::2]

        # Escalar las coordenadas de 0-1 a un tamaño de imagen arbitrario (ej. 1000x1000)
        x_points = [x * image_size for x in x_points]
        y_points = [y * image_size for y in y_points]

        # Generar un color aleatorio
        color = (random.random(), random.random(), random.random())

        # Dibujar el polígono
        plt.fill(x_points, y_points, color=color, alpha=0.5, edgecolor='black', linewidth=1)

    # Configurar el gráfico
    plt.xlim(0, image_size)
    plt.ylim(0, image_size)
    plt.gca().invert_yaxis()  # Invertir eje Y para que coincida con coordenadas de imagen
    plt.axis("equal")  # Mantener proporciones
    plt.title("Polígonos del Formato YOLO")
    plt.show()

# Ejemplo de uso
plot_yolo_polygons("./data/avocado-yolo-format/labels/2.txt")  # Reemplaza con la ruta de tu archivo YOLO