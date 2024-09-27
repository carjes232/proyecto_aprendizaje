import os
import time
import yaml

from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort

def load_config(config_path):
    """
    Carga el archivo de configuración YAML.

    Args:
        config_path (str): Ruta al archivo de configuración.

    Returns:
        dict: Configuración cargada.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"El archivo de configuración '{config_path}' no existe.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_image(image_path):
    """
    Carga la imagen original.

    Args:
        image_path (str): Ruta a la imagen de entrada.

    Returns:
        PIL.Image.Image: Imagen original.
        np.ndarray: Imagen original como arreglo de NumPy.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imagen '{image_path}' no existe.")
    
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    return image, image_np

def preprocess(image, input_size, keep_ratio):
    """
    Preprocesa la imagen para el modelo:
    - Redimensiona la imagen manteniendo la relación de aspecto si `keep_ratio` es True.
    - Rellena la imagen para que coincida con el tamaño de entrada si se mantiene la relación de aspecto.
    - Normaliza los valores de píxeles a [0,1].
    - Transpone las dimensiones a [C, H, W].
    - Agrega una dimensión de lote.

    Args:
        image (PIL.Image.Image): Imagen original.
        input_size (tuple): Tamaño de entrada deseado (ancho, alto).
        keep_ratio (bool): Si True, mantiene la relación de aspecto.

    Returns:
        np.ndarray: Tensor de imagen preprocesada.
        tuple: Tamaño original de la imagen (ancho, alto).
    """
    original_size = image.size  # (ancho, alto)
    target_width, target_height = input_size

    if keep_ratio:
        # Mantiene la relación de aspecto
        image.thumbnail((target_width, target_height), Image.BILINEAR)
        resized_width, resized_height = image.size
        pad_width = target_width - resized_width
        pad_height = target_height - resized_height
        # Rellena con color negro
        new_image = Image.new("RGB", input_size)
        new_image.paste(image, (pad_width // 2, pad_height // 2))
    else:
        # Redimensiona directamente
        new_image = image.resize(input_size, Image.BILINEAR)

    img_data = np.array(new_image).astype(np.float32) / 255.0  # Normaliza a [0,1]

    # Transpone a [C, H, W]
    img_data = img_data.transpose(2, 0, 1)

    # Agrega dimensión de lote
    img_tensor = np.expand_dims(img_data, axis=0)

    return img_tensor, original_size

def postprocess(outputs, original_size, config):
    """
    Postprocesa las salidas del modelo para extraer cajas delimitadoras, puntajes y IDs de clase.

    Args:
        outputs (list): Salidas crudas del modelo.
        original_size (tuple): Tamaño original de la imagen (ancho, alto).
        config (dict): Diccionario de configuración.

    Returns:
        list: Lista de detecciones con 'bbox', 'score', y 'class_id'.
    """
    # Extrae parámetros de configuración
    confidence_threshold = config.get('test_conf', 0.25)
    nms_threshold = config.get('nms_conf', 0.45)
    class_names = config.get('class_names', [])
    num_classes = config['model']['arch']['head']['num_classes']
    input_size = config['data']['val']['input_size']

    # Asumiendo que la salida del modelo es una lista con un solo arreglo [num_detecciones, 6]:
    # [x1, y1, x2, y2, score, class_id]
    output = outputs[0]  # Ajusta si tu modelo tiene múltiples salidas

    # Convierte a arreglo de NumPy si no lo es
    if not isinstance(output, np.ndarray):
        output = np.array(output)

    # Verifica la forma de la salida
    if output.ndim != 2 or output.shape[1] != 6:
        raise ValueError(f"La salida del modelo tiene una forma inesperada: {output.shape}. Se espera [num_detecciones, 6].")

    # Filtra detecciones con baja confianza
    boxes = output[:, :4]
    scores = output[:, 4]
    class_ids = output[:, 5].astype(int)

    indices = scores >= confidence_threshold
    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]

    # Aplica Supresión de No Máxima (NMS) por clase
    detections = []
    for cls in range(num_classes):
        cls_indices = np.where(class_ids == cls)[0]
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]

        if len(cls_boxes) == 0:
            continue

        # Convierte cajas y puntajes a listas para cv2.dnn.NMSBoxes
        cls_boxes_list = cls_boxes.tolist()
        cls_scores_list = cls_scores.tolist()

        # cv2.dnn.NMSBoxes espera cajas en el formato [x, y, width, height]
        cls_boxes_xywh = []
        for box in cls_boxes_list:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            cls_boxes_xywh.append([x1, y1, width, height])

        # Aplica NMS
        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=cls_boxes_xywh,
            scores=cls_scores_list,
            score_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )

        if len(nms_indices) > 0:
            for i in nms_indices.flatten():
                x, y, w, h = cls_boxes_xywh[i]
                score = cls_scores_list[i]
                bbox = [x, y, x + w, y + h]
                detections.append({
                    'bbox': bbox,
                    'score': float(score),
                    'class_id': int(cls),
                    'class_name': class_names[cls] if cls < len(class_names) else str(cls)
                })

    # Ajusta las cajas al tamaño original de la imagen
    resized_width, resized_height = input_size
    original_width, original_height = original_size
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    for det in detections:
        det['bbox'] = [
            int(det['bbox'][0] * scale_x),
            int(det['bbox'][1] * scale_y),
            int(det['bbox'][2] * scale_x),
            int(det['bbox'][3] * scale_y)
        ]

    return detections

def draw_detections(image_np, detections, confidence_threshold, output_path):
    """
    Dibuja cajas delimitadoras y etiquetas en la imagen.

    Args:
        image_np (np.ndarray): Imagen original como arreglo de NumPy.
        detections (list): Lista de detecciones con 'bbox', 'score' y 'class_id'.
        confidence_threshold (float): Umbral de confianza para mostrar detecciones.
        output_path (str): Ruta para guardar la imagen anotada.
    """
    annotated_image = image_np.copy()

    # Filtra detecciones basadas en el umbral de confianza
    filtered_detections = [d for d in detections if d['score'] >= confidence_threshold]
    print(f"Número de detecciones por encima del umbral ({confidence_threshold}): {len(filtered_detections)}")

    for idx, detection in enumerate(filtered_detections):
        bbox = detection['bbox']  # [x1, y1, x2, y2]
        score = detection['score']
        class_id = detection['class_id']
        class_name = detection['class_name']

        x1, y1, x2, y2 = bbox

        # Imprime detalles de la detección
        print(f"Detección {idx + 1}: Clase '{class_name}' (ID: {class_id}), Confianza {score:.2f}, Caja [{x1}, {y1}, {x2}, {y2}]")

        # Dibuja la caja delimitadora
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepara la etiqueta con la confianza
        label = f"{class_name}: {score:.2f}"

        # Calcula el tamaño del texto para el rectángulo de fondo
        (label_width, label_height), base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_height + 10)

        # Dibuja el rectángulo de fondo para la etiqueta
        cv2.rectangle(
            annotated_image,
            (x1, y1_label - label_height - 5),
            (x1 + label_width, y1_label + base_line - 5),
            (0, 255, 0),
            cv2.FILLED
        )

        # Coloca el texto de la etiqueta encima de la caja delimitadora
        cv2.putText(
            annotated_image,
            label,
            (x1, y1_label - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    # Guarda la imagen anotada usando OpenCV (convierte RGB a BGR)
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"Imagen anotada guardada en '{output_path}'")

def main():
    # Rutas de archivos
    config_path = "nanodet-m-0.5x.yml"
    onnx_model_path = "nanodet.onnx"
    image_path = "bus.jpg"
    output_image_path = "bus_detected.jpg"

    # Carga la configuración
    try:
        config = load_config(config_path)
        print(f"Configuración cargada desde '{config_path}'")
    except Exception as e:
        print(f"Error al cargar la configuración: {e}")
        return

    # Depuración: Imprimir las claves principales del config
    print("Claves principales en la configuración:", list(config.keys()))

    # Carga la imagen
    try:
        image, image_np = load_image(image_path)
        print(f"Imagen cargada desde '{image_path}'")
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return

    # Preprocesa la imagen
    try:
        input_size = tuple(config['data']['val']['input_size'])  # (ancho, alto)
        keep_ratio = config['data']['val'].get('keep_ratio', False)
        img_tensor, original_size = preprocess(image, input_size, keep_ratio)
        print(f"Imagen preprocesada a tamaño {input_size} con keep_ratio={keep_ratio}")
    except Exception as e:
        print(f"Error en el preprocesamiento de la imagen: {e}")
        return

    # Inicializa la sesión de ONNX Runtime
    if not os.path.exists(onnx_model_path):
        print(f"El modelo ONNX '{onnx_model_path}' no existe.")
        return

    try:
        # Puedes cambiar a 'CUDAExecutionProvider' si tienes una GPU compatible
        session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        print(f"Modelo ONNX cargado desde '{onnx_model_path}'")
    except Exception as e:
        print(f"Error al cargar el modelo ONNX: {e}")
        return

    # Obtiene el nombre de la entrada del modelo
    try:
        input_name = session.get_inputs()[0].name
        print(f"Nombre de la entrada del modelo: {input_name}")
    except Exception as e:
        print(f"Error al obtener el nombre de la entrada del modelo: {e}")
        return

    # Ejecuta la inferencia
    num_runs = 1  # Puedes ajustar el número de ejecuciones para medir el tiempo
    inference_times = []
    detections = []

    for i in range(num_runs):
        print(f"\nEjecución {i + 1}/{num_runs}")
        start_time = time.time()
        try:
            outputs = session.run(None, {input_name: img_tensor})
        except Exception as e:
            print(f"Error durante la inferencia: {e}")
            return
        end_time = time.time()

        inference_time = end_time - start_time
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
        inference_times.append(inference_time)

        # Para la última ejecución, procesa las detecciones
        if i == num_runs - 1:
            try:
                detections = postprocess(outputs, original_size, config)
            except Exception as e:
                print(f"Error en el postprocesamiento de las detecciones: {e}")
                return

    # Calcula el tiempo promedio de inferencia
    avg_inference_time = sum(inference_times) / num_runs
    print(f"\nTiempos de inferencia: {inference_times}")
    print(f"Tiempo promedio de inferencia: {avg_inference_time:.4f} segundos")

    # Dibuja y guarda las detecciones en la imagen si hay alguna detección
    if detections:
        try:
            draw_detections(
                image_np,
                detections,
                confidence_threshold=config.get('test_conf', 0.25),
                output_path=output_image_path
            )
        except Exception as e:
            print(f"Error al dibujar las detecciones: {e}")
    else:
        print("No se detectaron objetos con la confianza mínima especificada.")

if __name__ == "__main__":
    main()
