"""
Funciones auxiliares del TFG: Aplicación de técnicas de Inteligencia
Artificial para el cuidado de colmenas, detección de amenazas y plagas

Este módulo contiene las funciones desarrolladas durante el proyecto
para el procesamiento del dataset, entrenamiento/evaluación, inferencia
sobre vídeo, seguimiento de objetos y evaluación mediante SAHI.
"""

import os
import torch
import glob
import cv2
import shutil
import random
import time
import json
import yaml
import numpy as np
import textalloc as ta
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict
from torchvision import models, transforms

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou, ap_per_class

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image









def reparte_dataset_aleatorio(
        porcentaje_val,
        porcentaje_test,
        ruta_dataset,
        formato,
        seed=42):
    """
    Dado una carpeta de dataset con subcarpetas 'images/train' y 'labels/train', reparte aleatoriamente las imágenes y etiquetas en tres conjuntos: train, val y test.

    Parameters
    ----------
    porcentaje_val : float Porcentaje de imágenes que se destinarán al conjunto de validación (0 < porcentaje_val < 1).
    porcentaje_test : float Porcentaje de imágenes que se destinarán al conjunto de prueba (0 < porcentaje_test < 1).
    ruta_dataset : str Ruta a la carpeta del dataset.
    formato : str Formato de los archivos de imagen (por ejemplo, '.png').
    seed : int Semilla para la generación de números aleatorios.

    Warning
    ------
    Esta función mueve físicamente los archivos originales.
    """

    dataset_path = Path(ruta_dataset)

    if porcentaje_val + porcentaje_test >= 1:
        raise ValueError(
            "porcentaje_val + porcentaje_test debe ser menor que 1"
        )

    # Origen
    train_images = dataset_path / "images" / "train"
    train_labels = dataset_path / "labels" / "train"

    # Destinos
    val_images = dataset_path / "images" / "val"
    val_labels = dataset_path / "labels" / "val"

    test_images = dataset_path / "images" / "test"
    test_labels = dataset_path / "labels" / "test"

    # Crear carpetas
    for p in [
        val_images, val_labels,
        test_images, test_labels
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # Obtener imágenes
    image_files = list(train_images.glob("*" + formato))
    image_stems = [img.stem for img in image_files]

    random.seed(seed)
    random.shuffle(image_stems)

    total = len(image_stems)

    n_val = int(total * porcentaje_val)
    n_test = int(total * porcentaje_test)

    val_stems = image_stems[:n_val]
    test_stems = image_stems[n_val:n_val + n_test]

    # -------------------------
    # Mover VALIDACIÓN
    # -------------------------
    for stem in val_stems:

        img_src = train_images / f"{stem}{formato}"
        img_dst = val_images / f"{stem}{formato}"

        shutil.move(str(img_src), str(img_dst))

        label_src = train_labels / f"{stem}.txt"
        label_dst = val_labels / f"{stem}.txt"

        if label_src.exists():
            shutil.move(str(label_src), str(label_dst))

    # -------------------------
    # Mover TEST
    # -------------------------
    for stem in test_stems:

        img_src = train_images / f"{stem}{formato}"
        img_dst = test_images / f"{stem}{formato}"

        shutil.move(str(img_src), str(img_dst))

        label_src = train_labels / f"{stem}.txt"
        label_dst = test_labels / f"{stem}.txt"

        if label_src.exists():
            shutil.move(str(label_src), str(label_dst))

    print(f"✓ Total imágenes: {total}")
    print(f"✓ Train: {total - n_val - n_test}")
    print(f"✓ Val: {n_val}")
    print(f"✓ Test: {n_test}")




















def crop_dataset_to_center(folder_images, folder_labels, folder_output):
    """
    Recorta imágenes horizontales al cuadrado central y ajusta las etiquetas YOLO.
    
    Args:
        folder_images (str): Ruta a la carpeta con las imágenes .png.
        folder_labels (str): Ruta a la carpeta con los .txt de YOLO.
        folder_output (str): Ruta base donde se crearán las subcarpetas de salida.
    """
    
    # Crear la estructura de carpetas de salida
    out_images = os.path.join(folder_output, 'images')
    out_labels = os.path.join(folder_output, 'labels')
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # Iterar sobre las imágenes de la carpeta
    for img_name in os.listdir(folder_images):
        if not img_name.lower().endswith('.png'):
            continue

        # 1. Definir rutas individuales
        img_path = os.path.join(folder_images, img_name)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(folder_labels, txt_name)

        # 2. Leer imagen y calcular el recorte central
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ No se pudo leer la imagen: {img_path}")
            continue

        h_orig, w_orig, _ = img.shape
        
        # Como son horizontales, el cuadrado tendrá tamaño h_orig x h_orig
        # Calculamos desde qué píxel 'x' empieza el cuadrado central
        offset_x = (w_orig - h_orig) // 2
        
        # Recortar la imagen (en numpy/opencv es [y_start:y_end, x_start:x_end])
        cropped_img = img[:, offset_x : offset_x + h_orig]
        cv2.imwrite(os.path.join(out_images, img_name), cropped_img)

        # 3. Procesar las etiquetas YOLO
        new_labels = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = parts[0]
                x_c_norm, y_c_norm, w_norm, h_norm = map(float, parts[1:])

                # Convertir formato YOLO (normalizado) a píxeles absolutos originales
                x_c_abs = x_c_norm * w_orig
                y_c_abs = y_c_norm * h_orig
                w_abs = w_norm * w_orig
                h_abs = h_norm * h_orig

                # Obtener los 4 límites de la caja original
                xmin_orig = x_c_abs - w_abs / 2
                xmax_orig = x_c_abs + w_abs / 2
                ymin_orig = y_c_abs - h_abs / 2
                ymax_orig = y_c_abs + h_abs / 2

                # Límites del recorte (eje X)
                crop_xmin = offset_x
                crop_xmax = offset_x + h_orig

                # Si la caja está completamente fuera del recorte central, la ignoramos
                if xmax_orig <= crop_xmin or xmin_orig >= crop_xmax:
                    continue

                # Si está dentro (o parcialmente), ajustamos a las nuevas coordenadas locales
                new_xmin = max(0, xmin_orig - crop_xmin)
                new_xmax = min(h_orig, xmax_orig - crop_xmin)
                new_ymin = max(0, ymin_orig) 
                new_ymax = min(h_orig, ymax_orig)

                # Calcular el nuevo ancho y alto absolutos
                new_w_abs = new_xmax - new_xmin
                new_h_abs = new_ymax - new_ymin

                # Protección extra por si la caja se reduce a la nada
                if new_w_abs <= 0 or new_h_abs <= 0:
                    continue

                # Calcular el nuevo centro absoluto
                new_x_c_abs = new_xmin + new_w_abs / 2
                new_y_c_abs = new_ymin + new_h_abs / 2

                # Volver a normalizar para YOLO (ahora la imagen es h_orig x h_orig)
                new_x_c_norm = new_x_c_abs / h_orig
                new_y_c_norm = new_y_c_abs / h_orig
                new_w_norm = new_w_abs / h_orig
                new_h_norm = new_h_abs / h_orig

                # Guardar la línea formateada
                new_labels.append(f"{class_id} {new_x_c_norm:.6f} {new_y_c_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}")

        # 4. Guardar el nuevo fichero .txt (aunque esté vacío)
        out_txt_path = os.path.join(out_labels, txt_name)
        with open(out_txt_path, 'w') as f:
            if new_labels:
                f.write("\n".join(new_labels) + "\n")
                
    print(f"✅ Proceso terminado. Archivos generados en: {folder_output}")















def background_subtraction_and_mask(folder_input, folder_output, thresh_val=30):
    """
    Agrupa imágenes por plano, calcula el fondo promedio, aplica sustracción,
    limpia el ruido con operaciones morfológicas y usa el resultado como máscara
    para extraer los objetos en movimiento sobre fondo negro.
    
    Args:
        folder_input (str): Carpeta con las imágenes .png originales.
        folder_output (str): Carpeta base para guardar los resultados.
        thresh_val (int): Valor de umbral para separar ruido de movimiento (0-255).
    """
    planos_dict = defaultdict(list)
    
    if not os.path.exists(folder_input):
        print(f"❌ La carpeta de entrada no existe: {folder_input}")
        return

    # 1. Agrupar por plano
    for img_name in os.listdir(folder_input):
        if img_name.lower().endswith('.png'):
            if '_unido_' in img_name:
                plano_id = img_name.split('_unido_')[0]
            else:
                plano_id = img_name.split('_')[0]
            planos_dict[plano_id].append(img_name)

    print(f"🎬 Se han detectado {len(planos_dict)} planos diferentes.")

    # 2. Procesar cada plano
    for plano_id, img_list in planos_dict.items():
        print(f"⏳ Procesando plano: {plano_id} ({len(img_list)} imágenes)...")
        plano_out_folder = os.path.join(folder_output, plano_id)
        os.makedirs(plano_out_folder, exist_ok=True)

        # --- Calcular fondo promedio ---
        mean_background = None
        valid_img_count = 0

        for img_name in img_list:
            img_path = os.path.join(folder_input, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
                
            if mean_background is None:
                mean_background = img.astype(np.float64)
            else:
                mean_background += img.astype(np.float64)
            valid_img_count += 1

        if valid_img_count == 0: continue

        mean_background /= valid_img_count
        mean_background = np.clip(mean_background, 0, 255).astype(np.uint8)

        # --- Definir Kernels para Morfología ---
        kernel_noise = np.ones((3, 3), np.uint8)  # Para borrar puntos pequeños
        kernel_fill = np.ones((7, 7), np.uint8)   # Para rellenar huecos grandes del objeto

        # --- Sustracción, Morfología y Enmascarado ---
        for img_name in img_list:
            img_path = os.path.join(folder_input, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            diff = cv2.absdiff(img, mean_background)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_noise, iterations=1)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_fill, iterations=2)
            mask_clean = cv2.dilate(mask_clean, kernel_noise, iterations=3)
            final_result = cv2.bitwise_and(img, img, mask=mask_clean)

            # Guardar
            out_img_path = os.path.join(plano_out_folder, img_name)
            cv2.imwrite(out_img_path, final_result)

    print(f"\n✅ Proceso completado. Revisa la carpeta: {folder_output}")

def consolidar_imagenes_png(ruta_a):
    """
    Mueve todas las imágenes .png de las subcarpetas a una carpeta principal 'img'
    y luego elimina las subcarpetas.

    Warning
    -------
    La función elimina las subcarpetas originales después de mover las imágenes.
    """
    base_dir = Path(ruta_a)
    destino_dir = base_dir / 'img'
    
    # 1. Creamos la carpeta destino 'a/img' si no existe
    destino_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Iteramos por todos los elementos de la carpeta principal 'a'
    for subcarpeta in base_dir.iterdir():
        
        # Comprobamos que sea un directorio y evitamos procesar la propia carpeta destino 'img'
        if subcarpeta.is_dir() and subcarpeta != destino_dir:
            
            # 3. Buscamos todos los archivos .png dentro de la subcarpeta (incluso si están en subniveles)
            for img_path in subcarpeta.rglob('*.png'):
                destino_archivo = destino_dir / img_path.name
                
                # Movemos la imagen a la carpeta destino
                shutil.move(str(img_path), str(destino_archivo))
            
            # 4. Borramos la subcarpeta original y todo lo que quede dentro
            shutil.rmtree(subcarpeta)
            print(f"Subcarpeta eliminada: {subcarpeta.name}")

    print("\n¡Proceso completado! Todas las imágenes están ahora en:", destino_dir)

def background_subtraction_and_mask_labelMask(folder_input, folder_output, folder_txt=None, thresh_val=30):
    """
    Agrupa imágenes por plano, calcula el fondo promedio, aplica sustracción,
    limpia el ruido con operaciones morfológicas y usa el resultado junto con
    máscaras YOLO (opcional) para extraer los objetos en movimiento sobre fondo negro.
    
    Args:
        folder_input (str): Carpeta con las imágenes .png originales.
        folder_output (str): Carpeta base para guardar los resultados.
        folder_txt (str, optional): Carpeta con los .txt en formato YOLO. Por defecto None.
        thresh_val (int): Valor de umbral para separar ruido de movimiento (0-255).
    """
    planos_dict = defaultdict(list)
    
    if not os.path.exists(folder_input):
        print(f"❌ La carpeta de entrada no existe: {folder_input}")
        return

    # 1. Agrupar por plano
    for img_name in os.listdir(folder_input):
        if img_name.lower().endswith('.png'):
            if '_unido_' in img_name:
                plano_id = img_name.split('_unido_')[0]
            else:
                plano_id = img_name.split('_')[0]
            planos_dict[plano_id].append(img_name)

    print(f"🎬 Se han detectado {len(planos_dict)} planos diferentes.")

    # 2. Procesar cada plano
    for plano_id, img_list in planos_dict.items():
        print(f"⏳ Procesando plano: {plano_id} ({len(img_list)} imágenes)...")
        plano_out_folder = os.path.join(folder_output, plano_id)
        os.makedirs(plano_out_folder, exist_ok=True)

        # --- PASO A: Calcular fondo promedio ---
        mean_background = None
        valid_img_count = 0

        for img_name in img_list:
            img_path = os.path.join(folder_input, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
                
            if mean_background is None:
                mean_background = img.astype(np.float64)
            else:
                mean_background += img.astype(np.float64)
            valid_img_count += 1

        if valid_img_count == 0: continue

        mean_background /= valid_img_count
        mean_background = np.clip(mean_background, 0, 255).astype(np.uint8)

        # --- Definir Kernels para Morfología ---
        kernel_noise = np.ones((3, 3), np.uint8)  # Para borrar puntos pequeños
        kernel_fill = np.ones((7, 7), np.uint8)   # Para rellenar huecos grandes del objeto

        # --- PASO B: Sustracción, Morfología, YOLO y Enmascarado ---
        for img_name in img_list:
            img_path = os.path.join(folder_input, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            # 1. Máscara de Sustracción de Fondo
            diff = cv2.absdiff(img, mean_background)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_noise, iterations=1)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_fill, iterations=2)
            mask_clean = cv2.dilate(mask_clean, kernel_noise, iterations=3)

            # 2. Máscara de Bounding Boxes (YOLO)
            # Inicializamos la máscara YOLO completamente en negro (mismo tamaño)
            yolo_mask = np.zeros_like(mask_clean)

            if folder_txt is not None:
                txt_name = os.path.splitext(img_name)[0] + '.txt'
                txt_path = os.path.join(folder_txt, txt_name)
                
                if os.path.exists(txt_path):
                    h_img, w_img = img.shape[:2] # Obtenemos las dimensiones para desnormalizar
                    
                    with open(txt_path, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # YOLO: <clase> <x_center> <y_center> <width> <height>
                                _, x_c, y_c, w, h = map(float, parts[:5])
                                
                                # Desnormalizar coordenadas a píxeles
                                x_center_px = x_c * w_img
                                y_center_px = y_c * h_img
                                w_px = w * w_img
                                h_px = h * h_img
                                
                                # Calcular esquinas superior izquierda e inferior derecha
                                x_min = int(x_center_px - w_px / 2)
                                y_min = int(y_center_px - h_px / 2)
                                x_max = int(x_center_px + w_px / 2)
                                y_max = int(y_center_px + h_px / 2)
                                
                                # Dibujar el rectángulo en la máscara YOLO (255 = blanco)
                                cv2.rectangle(yolo_mask, (x_min, y_min), (x_max, y_max), 255, -1)

            # 3. Combinar ambas máscaras (OR lógico: nos quedamos con lo de ambas)
            combined_mask = cv2.bitwise_or(mask_clean, yolo_mask)

            # 4. Aplicar la máscara final a la imagen original
            final_result = cv2.bitwise_and(img, img, mask=combined_mask)

            # Guardar
            out_img_path = os.path.join(plano_out_folder, img_name)
            cv2.imwrite(out_img_path, final_result)

    print(f"\n✅ ¡Proceso completado con enmascaramiento combinado! Revisa la carpeta: {folder_output}")














def separar_FPs(model_path, images_dir, labels_dir, output_dir, text=True, reales=False, confi=0.25, iou_t=0.5, col_tp=(0, 255, 0), col_fp = (0, 0, 255), col_gt = (255, 0, 0), thick = 1 ):
    """
    Analiza imágenes con un modelo YOLO y separa las predicciones en Aciertos (TP) y Falsos Positivos (FP).

    Parameters
    ----------
    model_path : str Ruta al archivo de pesos del modelo YOLO.
    images_dir : str Carpeta que contiene las imágenes a analizar.
    labels_dir : str Carpeta que contiene los archivos de etiquetas en formato YOLO.
    output_dir : str Carpeta donde se guardarán las imágenes resultantes.
    text : bool Si se desea mostrar el texto con el nombre de la clase.
    reales : bool Si se desea dibujar las etiquetas reales.
    confi : float Confianza mínima para las predicciones.
    iou_t : float Umbral de solapamiento para considerar una predicción como acierto.
    col_tp : tuple Color para los Aciertos (True Positives).
    col_fp : tuple Color para los Falsos Positivos.
    col_gt : tuple Color para las Etiquetas Reales.
    thick : int Grosor de la línea.
    """
    conf_threshold = confi # Confianza mínima
    iou_threshold = iou_t  # Solapamiento mínimo
    color_tp = col_tp   # Verde para Aciertos (True Positives)
    color_fp = col_fp   # Rojo para Falsos Positivos
    color_gt = col_gt   # Azul para Etiquetas Reales (Opcional)
    thickness = thick   # Grosor de la línea
    os.makedirs(output_dir, exist_ok=True)
    
    print("Cargando modelo...")
    model = YOLO(model_path)
    
    print("Analizando imágenes y dibujando detecciones...")
    processed_count = 0
    
    # Obtener nombres de las clases del modelo para las etiquetas
    class_names = model.names
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): 
            continue
        
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        # Hacer predicción con el modelo
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        predictions = results[0].boxes
        h_img, w_img = results[0].orig_shape # Alto y ancho original
        
        # Cargar la imagen original para dibujar sobre ella
        img_cv2 = cv2.imread(img_path)
        
        # 1. Leer y dibujar Etiquetas Reales (Ground Truths) - Opcional, en AZUL
        ground_truths = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, x, y, w, h = map(float, parts)
                        gt_box = yolo_to_xyxy(x, y, w, h, w_img, h_img)
                        ground_truths.append({'class': int(c), 'box': gt_box})
                        if reales:
                            draw_box(img_cv2, gt_box, color_gt, f"GT:{class_names[int(c)]}", thickness, text)
        
        has_fp_to_save = False # Para no guardar imágenes sin FP
        
        # 2. Comprobar cada predicción del modelo y dibujar
        for pred in predictions:
            pred_class = int(pred.cls[0])
            pred_conf = pred.conf[0]
            pred_box = pred.xyxy[0].tolist()
            
            match_found = False
            for gt in ground_truths:
                # Si la clase es la misma y se solapan (IoU > 0.5), es un Acierto
                if gt['class'] == pred_class and calculate_iou(pred_box, gt['box']) > iou_threshold:
                    match_found = True
                    break
            
            # Determinar color y etiqueta según si es TP o FP
            if match_found:
                color = color_tp
                label = f"TP:{class_names[pred_class]} {pred_conf:.2f}"
            else:
                has_fp_to_save = True
                color = color_fp
                label = f"FP:{class_names[pred_class]} {pred_conf:.2f}"
                
            # Dibujar la caja y la etiqueta en la imagen
            draw_box(img_cv2, pred_box, color, label, thickness, text)
                
        # 3. Guardar la imagen si hubo alguna predicción (TP o FP)
        if has_fp_to_save:
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, img_cv2)
            processed_count += 1
    
    print(f"\n¡Listo! Se han generado {processed_count} imágenes visualizadas.")
    print(f"Puedes revisarlas en la carpeta: {output_dir}")














def generar_nuevas_etiquetas(model_path, images_dir, labels_dir, output_dir, conf_thresh=0.50, iou_thresh=0.20):
    """
    Genera nuevas etiquetas para imágenes en formato YOLO, combinando las predicciones del modelo con las etiquetas originales.

    Parameters
    ----------
    model_path : str Ruta al archivo de pesos del modelo YOLO.
    images_dir : str Carpeta que contiene las imágenes a analizar.
    labels_dir : str Carpeta que contiene los archivos de etiquetas en formato YOLO.
    output_dir : str Carpeta donde se guardarán las imágenes resultantes.
    conf_thresh : float Confianza mínima para las predicciones.
    iou_thresh : float Umbral de solapamiento para considerar una predicción como acierto.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    
    # Extensiones de imagen válidas
    imagenes = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]

    print(f"Procesando {len(imagenes)} imágenes...")

    for img_name in imagenes:
        img_path = os.path.join(images_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        orig_txt_path = os.path.join(labels_dir, txt_name)
        out_txt_path = os.path.join(output_dir, txt_name)

        original_boxes = []
        todas_las_lineas = []

        # 1. Leer las etiquetas originales (si la imagen ya tenía etiquetas)
        if os.path.exists(orig_txt_path):
            with open(orig_txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        todas_las_lineas.append(line.strip())
                        # Extraer solo [xc, yc, w, h] para calcular IoU después
                        original_boxes.append([float(x) for x in parts[1:]])

        # 2. Inferencia con el modelo
        results = model(img_path, conf=conf_thresh, verbose=False)

        # 3. Filtrar las predicciones
        for result in results:
            # result.boxes.xywhn devuelve cajas normalizadas igual que el formato YOLO
            for box, cls in zip(result.boxes.xywhn.tolist(), result.boxes.cls.tolist()):
                is_new = True
                # Comparar con todas las cajas originales para ver si ya existía
                for orig_box in original_boxes:
                    if compute_iou(box, orig_box) > iou_thresh:
                        is_new = False
                        break # Ya estaba etiquetada, la saltamos
                
                if is_new:
                    # Si es nueva, la añadimos a la lista final
                    nueva_linea = f"{int(cls)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                    todas_las_lineas.append(nueva_linea)

        # 4. Guardar el archivo combinado
        with open(out_txt_path, 'w') as f:
            for line in todas_las_lineas:
                f.write(f"{line}\n")

    print(f"¡Listo! Etiquetas combinadas guardadas en: {output_dir}")















def yolo_to_coco(images_dir, labels_dir, output_json, classes):
    """
        Convierte etiquetas en formato YOLO a formato COCO.
    
        Parameters
        ----------
        images_dir : str Carpeta que contiene las imágenes.
        labels_dir : str Carpeta que contiene los archivos de etiquetas en formato YOLO.
        output_json : str Ruta donde se guardará el archivo JSON en formato COCO.
        classes : list Lista de nombres de las clases.
    
        """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
    }
    
    annotation_id = 1
    image_id = 1
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # Forzar lectura del tamaño real de la imagen
        with Image.open(img_path) as img:
            width, height = img.size
            
        coco["images"].append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })
        
        # Si la imagen no tiene .txt, se procesa como imagen de fondo (Background)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                
                # Conversión: De YOLO normalizado a COCO absoluto
                w_abs = w * width
                h_abs = h * height
                x_min = (x_c - w / 2) * width
                y_min = (y_c - h / 2) * height
                
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, w_abs, h_abs],
                    "area": w_abs * h_abs,
                    "iscrowd": 0
                })
                annotation_id += 1
                
        image_id += 1
        
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=4)
    print(f"¡Éxito! Archivo COCO guardado en: {output_json}")

def coco_to_yolo(images_dir, labels_dir, class_mapping):
    """
    Convierte anotaciones JSON (estilo LabelMe) a formato YOLO (.txt).
    
    Args:
        images_dir (str): Ruta a la carpeta con las imágenes y los .json.
        labels_dir (str): Ruta a la carpeta donde se guardarán los .txt.
        class_mapping (dict): Diccionario que mapea nombres de clases a IDs numéricos.
    """
    # 1. Aseguramos que la carpeta de destino exista
    os.makedirs(labels_dir, exist_ok=True)
    
    # 2. Buscamos todos los archivos .json en la carpeta de imágenes
    json_files = glob.glob(os.path.join(images_dir, '*.json'))
    
    if not json_files:
        print(f"No se encontraron archivos .json en {images_dir}")
        return

    archivos_procesados = 0

    # 3. Iteramos sobre cada archivo JSON
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        img_w = data['imageWidth']
        img_h = data['imageHeight']
        
        # Obtenemos el nombre base sin extensión (ej: 'imagen_0009')
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        txt_path = os.path.join(labels_dir, base_name + '.txt')
        
        # 4. Abrimos el nuevo archivo .txt para escribir las detecciones
        with open(txt_path, 'w', encoding='utf-8') as f_out:
            for shape in data['shapes']:
                label = shape['label']
                
                # Si la clase no está en nuestro diccionario, la ignoramos o puedes lanzar error
                if label not in class_mapping:
                    print(f"⚠️ Clase '{label}' no encontrada en el mapping. Omitiendo...")
                    continue
                    
                class_id = class_mapping[label]
                points = shape['points']
                
                # 5. Extraer coordenadas extremas (funciona tenga 2 o 4 puntos el JSON)
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                
                # 6. Convertir a formato YOLO (normalizado)
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                
                # 7. Escribir la línea en el .txt
                f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
        archivos_procesados += 1
        
    print(f"✅ ¡Éxito! Se generaron {archivos_procesados} archivos .txt en {labels_dir}")
    print("Nota: Tus archivos .json originales siguen intactos en la carpeta de imágenes.")



















def benchmark_model(model, image, n_warmup=20, n_runs=100):
    """
        Mide el tiempo de inferencia de un modelo YOLO.

        Parameters
        ----------
        model : ultralytics.YOLO
            Modelo YOLO cargado.
        image : torch.Tensor
            Imagen de entrada.
        n_warmup : int, optional
            Número de iteraciones de warm-up.
        n_runs : int, optional
            Número de iteraciones para medir.

        Returns
        -------
        dict
            Estadísticas de tiempo en milisegundos y FPS.
        """
    # Warm-up: importante para que CUDA y el modelo estén preparados
    for _ in range(n_warmup):
        model.predict(
            image,
            device=0,
            verbose=False
        )

    torch.cuda.synchronize()

    times = []

    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        model.predict(
            image,
            device=0,
            verbose=False
        )

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

    times = np.array(times)

    return {
        "mean_ms": times.mean(),
        "median_ms": np.median(times),
        "std_ms": times.std(),
        "min_ms": times.min(),
        "max_ms": times.max(),
        "fps": 1000 / times.mean()
    }


def measure_vram(
    model,
    source,
    warmup=20,
):
    """
        Mide el uso de VRAM durante la inferencia de un modelo YOLO.

        Parameters
        ----------
        model : ultralytics.YOLO
            Modelo YOLO cargado.
        source : str
            Ruta a la imagen de entrada.
        warmup : int, optional
            Número de iteraciones de warm-up.

        Returns
        -------
        dict
            Estadísticas de uso de VRAM en megabytes.
        """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    device = torch.cuda.current_device()

    # Limpiar memoria no utilizada
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Warm-up
    for _ in range(warmup):
        model.predict(
            source=source,
            verbose=False,
        )

    # Sincronizar antes de medir
    torch.cuda.synchronize()

    # Reiniciar las estadísticas después del warm-up
    torch.cuda.reset_peak_memory_stats(device)

    # Inferencia que vamos a medir
    model.predict(
        source=source,
        verbose=False,
    )

    # Esperar a que CUDA termine
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    return {
        "allocated_mb": allocated / (1024 ** 2),
        "reserved_mb": reserved / (1024 ** 2),
        "peak_allocated_mb": peak_allocated / (1024 ** 2),
        "peak_reserved_mb": peak_reserved / (1024 ** 2),
    }

def benchmark_model_sahi(
    model_path,
    image,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    n_warmup=20,
    n_runs=100,
):
    """
    Mide el tiempo de inferencia de un modelo YOLO utilizando SAHI.

    Returns
    -------
    dict
        Estadísticas de tiempo en milisegundos y FPS.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    device = torch.cuda.current_device()

    # Cargar el modelo mediante SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.25,
        device=f"cuda:{device}",
    )

    # Warm-up
    for _ in range(n_warmup):
        get_sliced_prediction(
            image=image,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose=0,
        )

    torch.cuda.synchronize()

    times = []

    # Benchmark
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        get_sliced_prediction(
            image=image,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose=0,
        )

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

    times = np.array(times)

    return {
        "mean_ms": times.mean(),
        "median_ms": np.median(times),
        "std_ms": times.std(),
        "min_ms": times.min(),
        "max_ms": times.max(),
        "fps": 1000 / times.mean(),
    }


def analizar_video_yolo_completo(
    model,
    video_path,
    output_dir="resultados_video",
    save_every=30,
    verbose=True,
    conf=0.25,
    max_center_distance_ratio=0.08,
    max_missing=5,
    trace_length=60,
):
    """
    Analiza un vídeo completo con YOLO.

    Genera:
        1. Vídeo con detecciones, clases y confianza.
        2. Vídeo con fondo negro y rastros.
        3. Imágenes de determinados frames con:
           - imagen original
           - detecciones actuales
           - únicamente los rastros de las detecciones actuales

    Las imágenes se guardan cada 'save_every' frames.

    Parámetros
    ----------
    model : YOLO o str
        Modelo Ultralytics YOLO ya cargado o ruta al modelo.

    video_path : str
        Ruta al vídeo .mp4.

    output_dir : str
        Carpeta donde se guardarán todos los resultados.

    save_every : int
        Cada cuántos frames se guarda una imagen.
        Por ejemplo:
            save_every=30 -> guarda cada 30 frames.
            save_every=60 -> guarda cada 60 frames.

    verbose : bool
        Si True, imprime estadísticas al terminar.

    conf : float
        Confianza mínima de YOLO.

    max_center_distance_ratio : float
        Distancia máxima entre centros para asociar una detección
        a una trayectoria existente.

    max_missing : int
        Número máximo de frames consecutivos sin detección que
        se permite antes de terminar una trayectoria.

    trace_length : int
        Número máximo de posiciones que conserva cada trayectoria.

    Returns
    -------
    dict
        Rutas y estadísticas del análisis.
    """

    # ==============================================================
    # 1. Cargar modelo
    # ==============================================================

    if isinstance(model, str):
        model = YOLO(model)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"No existe el vídeo: {video_path}"
        )

    if save_every < 1:
        raise ValueError(
            "save_every debe ser >= 1."
        )

    os.makedirs(output_dir, exist_ok=True)

    # ==============================================================
    # 2. Abrir vídeo
    # ==============================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se ha podido abrir el vídeo: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    # ==============================================================
    # 3. Carpetas y nombres
    # ==============================================================

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    detections_path = os.path.join(
        output_dir,
        f"{video_name}_detections.mp4"
    )

    traces_path = os.path.join(
        output_dir,
        f"{video_name}_traces.mp4"
    )

    frames_dir = os.path.join(
        output_dir,
        f"{video_name}_frames"
    )

    os.makedirs(frames_dir, exist_ok=True)

    # ==============================================================
    # 4. VideoWriters
    # ==============================================================

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer_detections = cv2.VideoWriter(
        detections_path,
        fourcc,
        fps,
        (width, height)
    )

    writer_traces = cv2.VideoWriter(
        traces_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer_detections.isOpened():
        cap.release()
        raise RuntimeError(
            "No se pudo crear el vídeo de detecciones."
        )

    if not writer_traces.isOpened():
        cap.release()
        writer_detections.release()
        raise RuntimeError(
            "No se pudo crear el vídeo de rastros."
        )

    # ==============================================================
    # 5. Colores
    # ==============================================================

    # OpenCV utiliza BGR

    COLORS = {
        0: (0, 0, 255),      # orientalis -> rojo
        1: (255, 180, 0),    # bee -> azul/cian
    }

    CLASS_NAMES = {
        0: "orientalis",
        1: "bee",
    }

    # ==============================================================
    # 6. Parámetros del tracking
    # ==============================================================

    max_distance = (
        max_center_distance_ratio *
        min(width, height)
    )

    tracks = []

    next_track_id = 0

    class_switches = 0
    lost_detection_events = 0

    frames_saved = 0

    # ==============================================================
    # 7. Funciones auxiliares
    # ==============================================================

    def get_detections(result):
        """Extrae las detecciones de YOLO."""

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:

            xyxy = box.xyxy[0].cpu().numpy()

            confidence = float(
                box.conf[0].cpu().item()
            )

            class_id = int(
                box.cls[0].cpu().item()
            )

            if confidence < conf:
                continue

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": confidence,
                "class_id": class_id,
            })

        return detections

    def draw_detection(
        frame,
        detection,
        color=None
    ):
        """Dibuja una detección."""

        x1, y1, x2, y2 = detection["bbox"]

        cls = detection["class_id"]
        confidence = detection["confidence"]

        if color is None:
            color = COLORS.get(
                cls,
                (255, 255, 255)
            )

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        class_name = CLASS_NAMES.get(
            cls,
            str(cls)
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        # Tamaño del texto
        (
            text_w,
            text_h
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        # Fondo del texto
        cv2.rectangle(
            frame,
            (
                x1,
                max(
                    0,
                    y1 - text_h - baseline - 5
                )
            ),
            (
                x1 + text_w + 5,
                y1
            ),
            color,
            -1
        )

        # Texto
        cv2.putText(
            frame,
            label,
            (
                x1 + 2,
                y1 - 4
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    def draw_traces(
        frame,
        selected_tracks
    ):
        """
        Dibuja las trayectorias de los tracks recibidos.

        Para las imágenes de los frames seleccionados se le pasan
        únicamente los tracks que tienen una detección en ese frame.
        """

        for track in selected_tracks:

            history = track["history"]

            if len(history) < 2:
                continue

            start_index = max(
                0,
                len(history) - trace_length
            )

            history_to_draw = (
                history[start_index:]
            )

            for i in range(
                1,
                len(history_to_draw)
            ):

                p1, cls1 = (
                    history_to_draw[i - 1]
                )

                p2, cls2 = (
                    history_to_draw[i]
                )

                color = COLORS.get(
                    cls2,
                    (255, 255, 255)
                )

                cv2.line(
                    frame,
                    tuple(map(int, p1)),
                    tuple(map(int, p2)),
                    color,
                    2,
                    cv2.LINE_AA
                )

            # Punto actual
            center, cls = (
                history_to_draw[-1]
            )

            color = COLORS.get(
                cls,
                (255, 255, 255)
            )

            cv2.circle(
                frame,
                tuple(map(int, center)),
                4,
                color,
                -1
            )

    def center_distance(c1, c2):
        return np.sqrt(
            (c1[0] - c2[0]) ** 2
            +
            (c1[1] - c2[1]) ** 2
        )

    # ==============================================================
    # 8. Procesar vídeo
    # ==============================================================

    current_frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        current_frame_number += 1

        # ==========================================================
        # YOLO
        # ==========================================================

        result = model.predict(
            source=frame,
            conf=conf,
            verbose=False
        )[0]

        detections = get_detections(result)

        # ==========================================================
        # Asociar detecciones a tracks
        # ==========================================================

        matched_tracks = set()
        matched_detections = set()

        possible_matches = []

        for track_idx, track in enumerate(tracks):

            if not track["active"]:
                continue

            for det_idx, detection in enumerate(
                detections
            ):

                distance = center_distance(
                    track["last_center"],
                    detection["center"]
                )

                if distance <= max_distance:

                    possible_matches.append(
                        (
                            distance,
                            track_idx,
                            det_idx
                        )
                    )

        # Primero las asociaciones más cercanas
        possible_matches.sort(
            key=lambda x: x[0]
        )

        for (
            distance,
            track_idx,
            det_idx
        ) in possible_matches:

            if track_idx in matched_tracks:
                continue

            if det_idx in matched_detections:
                continue

            track = tracks[track_idx]

            detection = detections[det_idx]

            previous_class = (
                track["class_id"]
            )

            current_class = (
                detection["class_id"]
            )

            # ======================================================
            # Cambio de clase
            # ======================================================

            if previous_class != current_class:

                class_switches += 1

            # ======================================================
            # Actualizar trayectoria
            # ======================================================

            track["last_bbox"] = (
                detection["bbox"]
            )

            track["last_center"] = (
                detection["center"]
            )

            track["class_id"] = (
                current_class
            )

            track["confidence"] = (
                detection["confidence"]
            )

            track["history"].append(
                (
                    detection["center"],
                    current_class
                )
            )

            if len(track["history"]) > trace_length:
                track["history"].pop(0)

            track["missed"] = 0
            track["active"] = True

            matched_tracks.add(
                track_idx
            )

            matched_detections.add(
                det_idx
            )

        # ==========================================================
        # Tracks sin detección
        # ==========================================================

        for track_idx, track in enumerate(
            tracks
        ):

            if not track["active"]:
                continue

            if track_idx not in matched_tracks:

                # Solo contamos el inicio del episodio
                # de pérdida.
                if track["missed"] == 0:
                    lost_detection_events += 1

                track["missed"] += 1

                if track["missed"] > max_missing:
                    track["active"] = False

        # ==========================================================
        # Crear tracks nuevos
        # ==========================================================

        for det_idx, detection in enumerate(
            detections
        ):

            if det_idx in matched_detections:
                continue

            new_track = {
                "id": next_track_id,

                "last_bbox": detection["bbox"],

                "last_center": detection["center"],

                "class_id": detection["class_id"],

                "confidence": detection["confidence"],

                "missed": 0,

                "active": True,

                "history": [
                    (
                        detection["center"],
                        detection["class_id"]
                    )
                ],
            }

            tracks.append(new_track)

            next_track_id += 1

        # ==========================================================
        # VÍDEO DE DETECCIONES
        # ==========================================================

        detection_frame = frame.copy()

        for detection in detections:

            color = COLORS.get(
                detection["class_id"],
                (255, 255, 255)
            )

            draw_detection(
                detection_frame,
                detection,
                color
            )

        cv2.putText(
            detection_frame,
            f"Frame: {current_frame_number}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer_detections.write(
            detection_frame
        )

        # ==========================================================
        # VÍDEO DE RASTROS
        # ==========================================================

        trace_frame = np.zeros_like(frame)

        draw_traces(
            trace_frame,
            tracks
        )

        cv2.putText(
            trace_frame,
            f"Frame: {current_frame_number}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer_traces.write(
            trace_frame
        )

        # ==========================================================
        # GUARDAR IMAGEN
        #
        # Solo cada N frames.
        # ==========================================================

        if current_frame_number % save_every == 0:

            frame_image = frame.copy()

            # ------------------------------------------------------
            # IMPORTANTE:
            # Solo tracks que tienen detección EXACTAMENTE en
            # este frame.
            # ------------------------------------------------------

            active_tracks = [
                track
                for track in tracks
                if (
                    track["active"]
                    and track["missed"] == 0
                )
            ]

            # Dibujar únicamente esos rastros
            draw_traces(
                frame_image,
                active_tracks
            )

            # ------------------------------------------------------
            # Dibujar las detecciones actuales por encima
            # ------------------------------------------------------

            for detection in detections:

                color = COLORS.get(
                    detection["class_id"],
                    (255, 255, 255)
                )

                draw_detection(
                    frame_image,
                    detection,
                    color
                )

            # Número de frame
            cv2.putText(
                frame_image,
                f"Frame: {current_frame_number}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------------------
            # Guardar
            # ------------------------------------------------------

            frame_filename = os.path.join(
                frames_dir,
                f"frame_{current_frame_number:06d}.png"
            )

            cv2.imwrite(
                frame_filename,
                frame_image
            )

            frames_saved += 1

    # ==============================================================
    # 9. Cerrar
    # ==============================================================

    cap.release()

    writer_detections.release()
    writer_traces.release()

    # ==============================================================
    # 10. Resultados
    # ==============================================================

    results = {
        "detections_video": detections_path,
        "traces_video": traces_path,
        "frames_dir": frames_dir,

        "class_switches": class_switches,
        "lost_detection_events": lost_detection_events,

        "frames_saved": frames_saved,
        "total_frames": total_frames,

        "fps": fps,
        "width": width,
        "height": height,

        "save_every": save_every,
    }

    if verbose:

        print("\n========== RESULTADOS ==========")

        print(f"Vídeo: {video_path}")
        print(
            f"Resolución: {width}x{height}"
        )
        print(
            f"FPS: {fps:.2f}"
        )
        print(
            f"Frames totales: {total_frames}"
        )

        print()

        print(
            f"Cambios de clase: "
            f"{class_switches}"
        )

        print(
            f"Episodios de pérdida: "
            f"{lost_detection_events}"
        )

        print()

        print(
            f"Frames guardados: "
            f"{frames_saved}"
        )

        print(
            f"Frecuencia: cada "
            f"{save_every} frames"
        )

        print("\nArchivos:")

        print(
            f"  Detecciones: "
            f"{detections_path}"
        )

        print(
            f"  Rastros:     "
            f"{traces_path}"
        )

        print(
            f"  Imágenes:    "
            f"{frames_dir}"
        )

    return results


def analizar_video_yolo_completo_sahi(
    model,
    video_path,
    output_dir="resultados_video_sahi",
    save_every=30,
    verbose=True,
    conf=0.25,
    slice_size=400,
    overlap_ratio=0.2,
    device="cuda:0",
    batch_size=1,
    max_center_distance_ratio=0.08,
    max_missing=5,
    trace_length=60,
):
    """
    Analiza un vídeo completo utilizando SAHI + YOLO.

    Genera:

        1. Vídeo con detecciones, clases y confianza.
        2. Vídeo con fondo negro mostrando únicamente los rastros.
        3. Carpeta de imágenes de frames seleccionados.
           Cada imagen contiene:
               - frame original
               - detecciones actuales
               - únicamente las trayectorias de las detecciones
                 presentes en ese frame

    Además calcula:

        - Cambios de clase.
        - Episodios de pérdida de detección.

    Parámetros
    ----------
    model : ultralytics.YOLO, str
        Modelo YOLO cargado o ruta al .pt.

    video_path : str
        Ruta al vídeo.

    output_dir : str
        Carpeta de salida.

    save_every : int
        Guarda una imagen cada N frames.

    verbose : bool
        Imprime resultados al finalizar.

    conf : float
        Confianza mínima.

    slice_size : int
        Tamaño de cada slice de SAHI.
        Ejemplo: 400.

    overlap_ratio : float
        Solapamiento horizontal y vertical.
        Ejemplo: 0.2.

    device : str
        "cuda:0", "cpu", etc.

    batch_size : int
        Número de slices que SAHI procesa simultáneamente.

    max_center_distance_ratio : float
        Distancia máxima entre centros para asociar una detección
        con una trayectoria existente.

    max_missing : int
        Frames consecutivos que se toleran sin detección.

    trace_length : int
        Número máximo de posiciones almacenadas por trayectoria.

    Returns
    -------
    dict
        Rutas de salida y estadísticas.
    """

    # ==============================================================
    # 1. Validaciones
    # ==============================================================

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"No existe el vídeo: {video_path}"
        )

    if save_every < 1:
        raise ValueError(
            "save_every debe ser >= 1."
        )

    # ==============================================================
    # 2. Crear modelo SAHI
    # ==============================================================

    if isinstance(model, str):

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model,
            confidence_threshold=conf,
            device=device
        )

    else:

        # Si se pasa directamente un objeto YOLO
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model=model,
            confidence_threshold=conf,
            device=device
        )

    # ==============================================================
    # 3. Abrir vídeo
    # ==============================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se ha podido abrir el vídeo: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    # ==============================================================
    # 4. Directorios
    # ==============================================================

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    detections_path = os.path.join(
        output_dir,
        f"{video_name}_detections_sahi.mp4"
    )

    traces_path = os.path.join(
        output_dir,
        f"{video_name}_traces_sahi.mp4"
    )

    frames_dir = os.path.join(
        output_dir,
        f"{video_name}_frames_sahi"
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # ==============================================================
    # 5. VideoWriters
    # ==============================================================

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer_detections = cv2.VideoWriter(
        detections_path,
        fourcc,
        fps,
        (width, height)
    )

    writer_traces = cv2.VideoWriter(
        traces_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer_detections.isOpened():
        cap.release()
        raise RuntimeError(
            "No se pudo crear el vídeo de detecciones."
        )

    if not writer_traces.isOpened():
        cap.release()
        writer_detections.release()
        raise RuntimeError(
            "No se pudo crear el vídeo de rastros."
        )

    # ==============================================================
    # 6. Colores
    # ==============================================================

    # OpenCV = BGR

    COLORS = {
        0: (0, 0, 255),      # orientalis -> rojo
        1: (255, 180, 0),    # bee -> azul/cian
    }

    CLASS_NAMES = {
        0: "orientalis",
        1: "bee",
    }

    # ==============================================================
    # 7. Tracking
    # ==============================================================

    max_distance = (
        max_center_distance_ratio *
        min(width, height)
    )

    tracks = []

    next_track_id = 0

    class_switches = 0
    lost_detection_events = 0

    frames_saved = 0

    # ==============================================================
    # 8. Funciones auxiliares
    # ==============================================================

    def get_detections_sahi(result):
        """
        Convierte las predicciones SAHI al formato utilizado
        por el resto de la función.
        """

        detections = []

        for prediction in result.object_prediction_list:

            # ------------------------------------------------------
            # Bounding box
            # ------------------------------------------------------

            x1, y1, x2, y2 = (
                prediction.bbox.to_xyxy()
            )

            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            # ------------------------------------------------------
            # Confianza
            # ------------------------------------------------------

            confidence = float(
                prediction.score.value
            )

            if confidence < conf:
                continue

            # ------------------------------------------------------
            # Clase
            # ------------------------------------------------------

            class_id = int(
                prediction.category.id
            )

            # ------------------------------------------------------
            # Centro
            # ------------------------------------------------------

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detections.append({
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "center": (
                    cx,
                    cy
                ),
                "confidence": confidence,
                "class_id": class_id,
            })

        return detections

    def draw_detection(
        frame,
        detection,
        color=None
    ):
        """Dibuja una detección."""

        x1, y1, x2, y2 = detection["bbox"]

        class_id = detection["class_id"]
        confidence = detection["confidence"]

        if color is None:
            color = COLORS.get(
                class_id,
                (255, 255, 255)
            )

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        class_name = CLASS_NAMES.get(
            class_id,
            str(class_id)
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        # Tamaño del texto
        (
            text_w,
            text_h
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        # Fondo del texto
        cv2.rectangle(
            frame,
            (
                x1,
                max(
                    0,
                    y1 - text_h - baseline - 5
                )
            ),
            (
                x1 + text_w + 5,
                y1
            ),
            color,
            -1
        )

        # Texto
        cv2.putText(
            frame,
            label,
            (
                x1 + 2,
                y1 - 4
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    def draw_traces(
        frame,
        selected_tracks
    ):
        """
        Dibuja las trayectorias de los tracks recibidos.
        """

        for track in selected_tracks:

            history = track["history"]

            if len(history) < 2:
                continue

            start_index = max(
                0,
                len(history) - trace_length
            )

            history_to_draw = (
                history[start_index:]
            )

            for i in range(
                1,
                len(history_to_draw)
            ):

                p1, cls1 = (
                    history_to_draw[i - 1]
                )

                p2, cls2 = (
                    history_to_draw[i]
                )

                color = COLORS.get(
                    cls2,
                    (255, 255, 255)
                )

                cv2.line(
                    frame,
                    tuple(map(int, p1)),
                    tuple(map(int, p2)),
                    color,
                    2,
                    cv2.LINE_AA
                )

            # Punto actual
            center, cls = (
                history_to_draw[-1]
            )

            color = COLORS.get(
                cls,
                (255, 255, 255)
            )

            cv2.circle(
                frame,
                tuple(map(int, center)),
                4,
                color,
                -1
            )

    def center_distance(c1, c2):

        return np.sqrt(
            (c1[0] - c2[0]) ** 2
            +
            (c1[1] - c2[1]) ** 2
        )

    # ==============================================================
    # 9. Procesar vídeo
    # ==============================================================

    current_frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        current_frame_number += 1

        # ==========================================================
        # SAHI
        # ==========================================================

        # OpenCV entrega BGR.
        # SAHI utiliza la imagen en formato RGB.
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = get_sliced_prediction(
            frame_rgb,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            #batch_size=batch_size,
            verbose=0,
        )

        # ==========================================================
        # Extraer detecciones SAHI
        # ==========================================================

        detections = get_detections_sahi(
            result
        )

        # ==========================================================
        # Asociar detecciones a tracks
        # ==========================================================

        matched_tracks = set()
        matched_detections = set()

        possible_matches = []

        for track_idx, track in enumerate(
            tracks
        ):

            if not track["active"]:
                continue

            for det_idx, detection in enumerate(
                detections
            ):

                distance = center_distance(
                    track["last_center"],
                    detection["center"]
                )

                if distance <= max_distance:

                    possible_matches.append(
                        (
                            distance,
                            track_idx,
                            det_idx
                        )
                    )

        # Asociar primero las detecciones
        # espacialmente más cercanas.
        possible_matches.sort(
            key=lambda x: x[0]
        )

        for (
            distance,
            track_idx,
            det_idx
        ) in possible_matches:

            if track_idx in matched_tracks:
                continue

            if det_idx in matched_detections:
                continue

            track = tracks[track_idx]

            detection = detections[det_idx]

            previous_class = (
                track["class_id"]
            )

            current_class = (
                detection["class_id"]
            )

            # ======================================================
            # Cambio de clase
            # ======================================================

            if previous_class != current_class:

                class_switches += 1

            # ======================================================
            # Actualizar track
            # ======================================================

            track["last_bbox"] = (
                detection["bbox"]
            )

            track["last_center"] = (
                detection["center"]
            )

            track["class_id"] = (
                current_class
            )

            track["confidence"] = (
                detection["confidence"]
            )

            track["history"].append(
                (
                    detection["center"],
                    current_class
                )
            )

            if len(track["history"]) > trace_length:
                track["history"].pop(0)

            track["missed"] = 0
            track["active"] = True

            matched_tracks.add(
                track_idx
            )

            matched_detections.add(
                det_idx
            )

        # ==========================================================
        # Tracks sin detección
        # ==========================================================

        for track_idx, track in enumerate(
            tracks
        ):

            if not track["active"]:
                continue

            if track_idx not in matched_tracks:

                # Solo contabilizamos el inicio de la pérdida.
                if track["missed"] == 0:
                    lost_detection_events += 1

                track["missed"] += 1

                if track["missed"] > max_missing:
                    track["active"] = False

        # ==========================================================
        # Crear tracks nuevos
        # ==========================================================

        for det_idx, detection in enumerate(
            detections
        ):

            if det_idx in matched_detections:
                continue

            new_track = {
                "id": next_track_id,

                "last_bbox": detection["bbox"],

                "last_center": detection["center"],

                "class_id": detection["class_id"],

                "confidence": detection["confidence"],

                "missed": 0,

                "active": True,

                "history": [
                    (
                        detection["center"],
                        detection["class_id"]
                    )
                ],
            }

            tracks.append(
                new_track
            )

            next_track_id += 1

        # ==========================================================
        # VÍDEO DE DETECCIONES
        # ==========================================================

        detection_frame = frame.copy()

        for detection in detections:

            color = COLORS.get(
                detection["class_id"],
                (255, 255, 255)
            )

            draw_detection(
                detection_frame,
                detection,
                color
            )

        cv2.putText(
            detection_frame,
            f"Frame: {current_frame_number}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer_detections.write(
            detection_frame
        )

        # ==========================================================
        # VÍDEO DE RASTROS
        # ==========================================================

        trace_frame = np.zeros_like(
            frame
        )

        draw_traces(
            trace_frame,
            tracks
        )

        cv2.putText(
            trace_frame,
            f"Frame: {current_frame_number}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer_traces.write(
            trace_frame
        )

        # ==========================================================
        # GUARDAR FRAME CADA N
        # ==========================================================

        if current_frame_number % save_every == 0:

            frame_image = frame.copy()

            # ======================================================
            # SOLO tracks detectados EN ESTE FRAME
            # ======================================================

            active_tracks = [
                track
                for track in tracks
                if (
                    track["active"]
                    and track["missed"] == 0
                )
            ]

            # Rastros
            draw_traces(
                frame_image,
                active_tracks
            )

            # ======================================================
            # Detecciones actuales
            # ======================================================

            for detection in detections:

                color = COLORS.get(
                    detection["class_id"],
                    (255, 255, 255)
                )

                draw_detection(
                    frame_image,
                    detection,
                    color
                )

            # Número del frame
            cv2.putText(
                frame_image,
                f"Frame: {current_frame_number}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Guardar
            frame_filename = os.path.join(
                frames_dir,
                f"frame_{current_frame_number:06d}.png"
            )

            cv2.imwrite(
                frame_filename,
                frame_image
            )

            frames_saved += 1

    # ==============================================================
    # 10. Cerrar
    # ==============================================================

    cap.release()

    writer_detections.release()
    writer_traces.release()

    # ==============================================================
    # 11. Resultados
    # ==============================================================

    results = {
        "detections_video": detections_path,
        "traces_video": traces_path,
        "frames_dir": frames_dir,

        "class_switches": class_switches,
        "lost_detection_events": lost_detection_events,

        "frames_saved": frames_saved,
        "total_frames": total_frames,

        "fps": fps,
        "width": width,
        "height": height,

        "save_every": save_every,

        "slice_size": slice_size,
        "overlap_ratio": overlap_ratio,
        "conf": conf,
    }

    if verbose:

        print("\n========== RESULTADOS SAHI ==========")

        print(f"Vídeo: {video_path}")
        print(
            f"Resolución: "
            f"{width}x{height}"
        )
        print(
            f"FPS: {fps:.2f}"
        )
        print(
            f"Frames totales: "
            f"{total_frames}"
        )

        print("\n--- SAHI ---")

        print(
            f"Slice: "
            f"{slice_size}x{slice_size}"
        )

        print(
            f"Overlap: "
            f"{overlap_ratio}"
        )

        print(
            f"Confidence: "
            f"{conf}"
        )

        print("\n--- Seguimiento ---")

        print(
            f"Cambios de clase: "
            f"{class_switches}"
        )

        print(
            f"Episodios de pérdida: "
            f"{lost_detection_events}"
        )

        print("\n--- Frames ---")

        print(
            f"Frames guardados: "
            f"{frames_saved}"
        )

        print(
            f"Frecuencia: cada "
            f"{save_every} frames"
        )

        print("\n--- Archivos ---")

        print(
            f"Detecciones: "
            f"{detections_path}"
        )

        print(
            f"Rastros:     "
            f"{traces_path}"
        )

        print(
            f"Imágenes:    "
            f"{frames_dir}"
        )

    return results

def val_yolo_sahi(
    ruta_datos_yaml,
    nombre,
    slice_size=400,
    overlap_ratio=0.2,
    imgsz=640,
    conf=0.001,
    device="cuda:0",
    conjunto="val",
):
    """
    Evalúa un modelo YOLO sobre el conjunto 'val' usando SAHI.

    La imagen completa se divide en slices con SAHI y las detecciones
    se vuelven a proyectar sobre la imagen original.

    Las métricas finales se calculan con ap_per_class() de Ultralytics,
    por lo que son comparables con las métricas habituales de YOLO.

    Parámetros
    ----------
    ruta_datos_yaml : str
        Ruta al dataset.yaml.

    nombre : str
        Nombre de la carpeta del entrenamiento.
        Ejemplo:
            nombre = "mi_modelo"

        Se cargará:
            runs/detect/mi_modelo/weights/best.pt

    slice_size : int
        Tamaño de cada slice.

    overlap_ratio : float
        Solapamiento entre slices.

    imgsz : int
        Tamaño de entrada de YOLO en cada slice.

    conf : float
        Umbral mínimo de confianza de SAHI.
        Para evaluación de mAP conviene mantenerlo bajo.

    device : str
        "cuda:0" o "cpu".

    Returns
    -------
    dict
        Diccionario con las métricas globales y por clase.
    """

    # ============================================================
    # 1. CARGAR YAML
    # ============================================================

    ruta_datos_yaml = Path(ruta_datos_yaml)

    with open(ruta_datos_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, list):
        names = {
            i: name
            for i, name in enumerate(names)
        }

    nc = len(names)

    # ============================================================
    # 2. RESOLVER "path" DEL DATASET
    # ============================================================

    dataset_root = Path(data["path"])

    # Primero intentamos exactamente como está escrito en el YAML,
    # relativo al directorio de trabajo del notebook.
    if not dataset_root.is_absolute():
        if dataset_root.exists():
            pass
        else:
            # Como segunda opción, relativo a la carpeta del YAML.
            dataset_root = (
                ruta_datos_yaml.parent / dataset_root
            )

    dataset_root = dataset_root.resolve()

    val_relative = Path(data[conjunto])
    val_dir = dataset_root / val_relative

    if not val_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de validación:\n{val_dir}\n"
            f"\nDataset root detectado:\n{dataset_root}"
        )

    # ============================================================
    # 3. OBTENER IMÁGENES DE VALIDACIÓN
    # ============================================================

    extensiones = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    image_paths = sorted(
        [
            p
            for p in val_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in extensiones
        ]
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No se encontraron imágenes en:\n{val_dir}"
        )

    print(f"Dataset: {dataset_root}")
    print(f"Validación: {val_dir}")
    print(f"Imágenes encontradas: {len(image_paths)}")

    # ============================================================
    # 4. MODELO
    # ============================================================

    model_path = (
        Path("runs")
        / "detect"
        / nombre
        / "weights"
        / "best.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo:\n{model_path}"
        )

    print(f"Modelo: {model_path}")
    print(
        f"SAHI: {slice_size}x{slice_size} "
        f"overlap={overlap_ratio}"
    )

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=conf,
        device=device,
        image_size=imgsz,
    )

    # ============================================================
    # 5. ACUMULADORES
    # ============================================================

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []

    total_instances = 0
    total_images = len(image_paths)

    total_time = 0.0

    # ============================================================
    # 6. RECORRER VALIDACIÓN
    # ============================================================

    for image_idx, image_path in enumerate(image_paths):

        print(
            f"\rVal: {image_idx + 1}/{total_images}",
            end=""
        )

        # --------------------------------------------------------
        # LEER IMAGEN
        # --------------------------------------------------------

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # --------------------------------------------------------
        # LABEL
        # --------------------------------------------------------

        relative_image = image_path.relative_to(
            dataset_root
        )

        # images/val/xxxx.jpg
        # ->
        # labels/val/xxxx.txt

        relative_label = Path(
            str(relative_image).replace(
                "images",
                "labels",
                1,
            )
        ).with_suffix(".txt")

        label_path = dataset_root / relative_label

        gt_boxes = []
        gt_classes = []

        if label_path.exists():

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    values = line.strip().split()

                    if len(values) < 5:
                        continue

                    cls = int(values[0])

                    x_center = float(values[1])
                    y_center = float(values[2])
                    width = float(values[3])
                    height = float(values[4])

                    # YOLO normalizado -> xyxy
                    x1 = (
                        x_center - width / 2
                    ) * image_width

                    y1 = (
                        y_center - height / 2
                    ) * image_height

                    x2 = (
                        x_center + width / 2
                    ) * image_width

                    y2 = (
                        y_center + height / 2
                    ) * image_height

                    gt_boxes.append(
                        [x1, y1, x2, y2]
                    )

                    gt_classes.append(cls)

        gt_classes = np.asarray(
            gt_classes,
            dtype=np.float32,
        )

        gt_boxes = np.asarray(
            gt_boxes,
            dtype=np.float32,
        ).reshape(-1, 4)

        total_instances += len(gt_classes)

        # --------------------------------------------------------
        # INFERENCIA SAHI
        # --------------------------------------------------------

        start = time.perf_counter()

        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
            #progress_bar=False,
        )

        total_time += (
            time.perf_counter() - start
        )

        # --------------------------------------------------------
        # EXTRAER PREDICCIONES
        # --------------------------------------------------------

        pred_boxes = []
        pred_conf = []
        pred_classes = []

        for prediction in (
            result.object_prediction_list
        ):

            pred_boxes.append(
                prediction.bbox.to_xyxy()
            )

            pred_conf.append(
                float(
                    prediction.score.value
                )
            )

            pred_classes.append(
                int(
                    prediction.category.id
                )
            )

        pred_boxes = np.asarray(
            pred_boxes,
            dtype=np.float32,
        ).reshape(-1, 4)

        pred_conf = np.asarray(
            pred_conf,
            dtype=np.float32,
        )

        pred_classes = np.asarray(
            pred_classes,
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # TP
        # --------------------------------------------------------

        n_pred = len(pred_boxes)

        tp = np.zeros(
            (n_pred, 10),
            dtype=bool,
        )

        if (
            n_pred > 0
            and len(gt_boxes) > 0
        ):

            pred_boxes_t = torch.from_numpy(
                pred_boxes
            ).to(device)

            gt_boxes_t = torch.from_numpy(
                gt_boxes
            ).to(device)

            pred_classes_t = torch.from_numpy(
                pred_classes
            ).to(device)

            gt_classes_t = torch.from_numpy(
                gt_classes
            ).to(device)

            # [GT, predictions]
            ious = box_iou(
                gt_boxes_t,
                pred_boxes_t
            )

            # Solo permitimos match de la misma clase
            class_match = (
                gt_classes_t[:, None]
                == pred_classes_t[None, :]
            )

            ious[~class_match] = 0

            # Ordenar predicciones por confianza
            order = np.argsort(
                -pred_conf
            )

            # Convertimos para trabajar ordenadas
            ious = ious[:, order]

            # IoU 0.50 ... 0.95
            thresholds = np.arange(
                0.50,
                1.00,
                0.05,
            )

            for t_idx, threshold in enumerate(
                thresholds
            ):

                used_gt = set()
                used_pred = set()

                # Buscar candidatos
                candidates = torch.where(
                    ious > threshold
                )

                if len(candidates[0]) == 0:
                    continue

                candidate_ious = ious[
                    candidates[0],
                    candidates[1],
                ]

                # Ordenar por IoU descendente
                order_candidates = torch.argsort(
                    candidate_ious,
                    descending=True,
                )

                for k in order_candidates:

                    gt_idx = int(
                        candidates[0][k]
                    )

                    pred_idx_sorted = int(
                        candidates[1][k]
                    )

                    if gt_idx in used_gt:
                        continue

                    if pred_idx_sorted in used_pred:
                        continue

                    used_gt.add(gt_idx)
                    used_pred.add(
                        pred_idx_sorted
                    )

                    original_pred_idx = order[
                        pred_idx_sorted
                    ]

                    tp[
                        original_pred_idx,
                        t_idx,
                    ] = True

        # --------------------------------------------------------
        # GUARDAR RESULTADOS
        # --------------------------------------------------------

        if n_pred > 0:

            all_tp.append(tp)
            all_conf.append(pred_conf)
            all_pred_cls.append(pred_classes)

        # Aunque no haya predicciones, los GT deben entrar
        # en target_cls para contabilizar FN.
        if len(gt_classes) > 0:
            all_target_cls.append(
                gt_classes
            )

    print()

    # ============================================================
    # 7. CONCATENAR
    # ============================================================

    if all_tp:

        all_tp = np.concatenate(
            all_tp,
            axis=0,
        )

        all_conf = np.concatenate(
            all_conf,
            axis=0,
        )

        all_pred_cls = np.concatenate(
            all_pred_cls,
            axis=0,
        )

    else:

        all_tp = np.zeros(
            (0, 10),
            dtype=bool,
        )

        all_conf = np.empty(
            0,
            dtype=np.float32,
        )

        all_pred_cls = np.empty(
            0,
            dtype=np.float32,
        )

    if all_target_cls:

        all_target_cls = np.concatenate(
            all_target_cls,
            axis=0,
        )

    else:

        all_target_cls = np.empty(
            0,
            dtype=np.float32,
        )

    # ============================================================
    # 8. MÉTRICAS ULTRALYTICS
    # ============================================================

    if (
        len(all_pred_cls) == 0
        and len(all_target_cls) == 0
    ):
        raise RuntimeError(
            "No hay ni predicciones ni ground truth."
        )

    (
        tp_best,
        fp_best,
        precision,
        recall,
        f1,
        ap,
        ap_class_index,
        *_
    ) = ap_per_class(
        all_tp,
        all_conf,
        all_pred_cls,
        all_target_cls,
        names=names,
    )

    # ============================================================
    # 9. RESULTADOS
    # ============================================================

    # mAP50
    map50_per_class = ap[:, 0]

    # mAP50-95
    map5095_per_class = ap.mean(
        axis=1
    )

    print(
        f"\n{'Class':<15}"
        f"{'Images':>10}"
        f"{'Instances':>12}"
        f"{'Box(P)':>10}"
        f"{'R':>10}"
        f"{'mAP50':>10}"
        f"{'mAP50-95':>12}"
    )

    # ------------------------------------------------------------
    # POR CLASE
    # ------------------------------------------------------------

    class_results = {}

    for i, cls_idx in enumerate(
        ap_class_index
    ):

        cls_idx = int(cls_idx)

        class_name = names[cls_idx]

        instances = int(
            np.sum(
                all_target_cls == cls_idx
            )
        )

        p = precision[i]
        r = recall[i]
        map50 = map50_per_class[i]
        map5095 = map5095_per_class[i]

        class_results[
            class_name
        ] = {
            "precision": float(p),
            "recall": float(r),
            "map50": float(map50),
            "map50_95": float(map5095),
            "instances": instances,
        }

        print(
            f"{class_name:<15}"
            f"{'--':>10}"
            f"{instances:>12}"
            f"{p:>10.3f}"
            f"{r:>10.3f}"
            f"{map50:>10.3f}"
            f"{map5095:>12.3f}"
        )

    # ------------------------------------------------------------
    # GLOBAL
    # ------------------------------------------------------------

    precision_global = np.mean(
        precision
    )

    recall_global = np.mean(
        recall
    )

    map50_global = np.mean(
        map50_per_class
    )

    map5095_global = np.mean(
        map5095_per_class
    )

    print(
        f"{'all':<15}"
        f"{total_images:>10}"
        f"{total_instances:>12}"
        f"{precision_global:>10.3f}"
        f"{recall_global:>10.3f}"
        f"{map50_global:>10.3f}"
        f"{map5095_global:>12.3f}"
    )

    # ============================================================
    # 10. TIEMPO
    # ============================================================

    avg_time_ms = (
        total_time
        / total_images
        * 1000
    )

    print(
        f"\nSAHI inference: "
        f"{avg_time_ms:.2f} ms/image"
    )

    print(
        f"FPS equivalente: "
        f"{1000 / avg_time_ms:.2f}"
    )

    print("Predicciones:", len(all_conf))
    print("GT:", len(all_target_cls))

    # ============================================================
    # 11. RETURN
    # ============================================================

    return {
        "precision": precision_global,
        "recall": recall_global,
        "map50": map50_global,
        "map50_95": map5095_global,
        "per_class": class_results,
        "tp": all_tp,
        "conf": all_conf,
        "pred_cls": all_pred_cls,
        "target_cls": all_target_cls,
        "inference_time_ms": avg_time_ms,
        "fps": 1000 / avg_time_ms,
    }














# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def yolo_to_xyxy(x_c, y_c, w, h, img_w, img_h):
    """Convierte formato YOLO a coordenadas de caja [x1, y1, x2, y2]"""
    return [
        (x_c - w/2) * img_w, (y_c - h/2) * img_h,
        (x_c + w/2) * img_w, (y_c + h/2) * img_h
    ]

def calculate_iou(boxA, boxB):
    """Calcula el Intersection over Union (IoU) entre dos cajas"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

def draw_box(img, box, color, label, thickness=2, text=True):
    """Dibuja una caja y una etiqueta en la imagen"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,thickness)

def compute_iou(box1, box2):
    """Calcula el IoU entre dos cajas en formato YOLO [x_center, y_center, width, height]"""
    # Convertir a coordenadas [x1, y1, x2, y2]
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    # Calcular área de intersección
    x_left = max(b1_x1, b2_x1)
    y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2)
    y_bottom = min(b1_y2, b2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = (box1[2]*box1[3]) + (box2[2]*box2[3]) - intersection
    return intersection / union if union > 0 else 0.0