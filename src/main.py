import argparse
import os
import sys
import time
import types

import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Предупреждение: MediaPipe не установлен. Установите: pip install mediapipe")


def fix_image_orientation(img: Image.Image) -> Image.Image:
    """
    Исправляет ориентацию изображения на основе EXIF данных.
    Многие камеры сохраняют изображения с метаданными ориентации.
    Использует ImageOps.exif_transpose для автоматической коррекции.
    """
    try:
        # Современный способ (PIL 8.0+)
        img = ImageOps.exif_transpose(img)
    except (AttributeError, TypeError, ValueError):
        # Если ImageOps.exif_transpose недоступен, используем старый способ
        try:
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                orientation = exif.get(274)  # EXIF tag 274 = Orientation
                
                if orientation == 2:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 4:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                elif orientation == 5:
                    img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90, expand=True)
                elif orientation == 7:
                    img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, TypeError):
            # Если EXIF данных нет или ошибка при чтении, просто возвращаем изображение как есть
            pass
    
    return img


def normalize_exposure(img: Image.Image, debug: bool = False) -> Image.Image:
    """
    Нормализует экспозицию изображения (яркость и контраст).
    Использует CLAHE (Contrast Limited Adaptive Histogram Equalization) для улучшения
    и дополнительное повышение яркости.
    """
    if img.mode != "RGB":
        img_rgb = img.convert("RGB")
    else:
        img_rgb = img.copy()
    
    # Конвертируем в numpy array для OpenCV
    img_array = np.array(img_rgb)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Применяем CLAHE к каналу L (яркость) с увеличенным clipLimit для большей яркости
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    img_cv[:, :, 0] = clahe.apply(img_cv[:, :, 0])
    
    # Дополнительно повышаем яркость (увеличиваем канал L на 10-15%)
    l_channel = img_cv[:, :, 0].astype(np.float32)
    l_channel = l_channel * 1.15  # Увеличиваем яркость на 15%
    l_channel = np.clip(l_channel, 0, 255).astype(np.uint8)
    img_cv[:, :, 0] = l_channel
    
    # Конвертируем обратно в RGB
    img_normalized = cv2.cvtColor(img_cv, cv2.COLOR_LAB2RGB)
    
    # Конвертируем обратно в PIL Image
    result = Image.fromarray(img_normalized)
    
    # Сохраняем альфа-канал, если был
    if img.mode == "RGBA":
        result = result.convert("RGBA")
        alpha = img.split()[3]
        result.putalpha(alpha)
    
    if debug:
        print("📸 Нормализация экспозиции применена (CLAHE + повышение яркости на 15%)")
    
    return result


def load_image(path: str, debug_orientation: bool = False) -> Image.Image:
    """
    Загружает изображение в формате RGBA с исправлением ориентации.
    
    Args:
        path: Путь к изображению
        debug_orientation: Если True, выводит информацию о коррекции ориентации
    """
    img = Image.open(path)
    original_size = img.size
    
    # Исправляем ориентацию перед конвертацией
    img = fix_image_orientation(img)
    
    if debug_orientation and img.size != original_size:
        print(f"⚠️ Ориентация изображения была исправлена: {original_size} -> {img.size}")
    
    img = img.convert("RGBA")
    return img


def detect_face_mediapipe(img: Image.Image | np.ndarray, debug: bool = True, debug_file: str | None = None) -> dict | None:
    """
    Распознает лицо с помощью MediaPipe и возвращает информацию о нем.
    
    Args:
        img: Изображение в формате PIL Image или numpy array (BGR/RGB)
        debug: Если True, выводит дебаг информацию в консоль
        debug_file: Если указан путь к файлу, сохраняет дебаг информацию в файл
    
    Returns:
        Словарь с информацией о лице:
        - 'detected': bool - обнаружено ли лицо
        - 'face_count': int - количество обнаруженных лиц
        - 'landmarks': list - список landmarks (468 точек для Face Mesh)
        - 'bounding_boxes': list - список bounding boxes для каждого лица
        - 'face_landmarks_2d': list - 2D координаты landmarks
        - 'face_landmarks_3d': list - 3D координаты landmarks (если доступны)
        - 'face_blendshapes': list - blendshapes (если доступны)
        - 'face_geometry': dict - геометрия лица (если доступна)
    """
    # Функция для вывода (в консоль и/или файл)
    debug_output = []
    def debug_print(*args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        if debug:
            print(*args, **kwargs)
        if debug_file:
            debug_output.append(msg)
    
    if not MEDIAPIPE_AVAILABLE:
        if debug:
            debug_print("❌ MediaPipe не установлен. Пропускаем распознавание лица.")
        return None
    
    # Конвертируем PIL Image в numpy array если нужно
    if isinstance(img, Image.Image):
        img_array = np.array(img.convert("RGB"))
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_rgb = img.copy()
        if len(img_rgb.shape) == 4:  # RGBA
            img_rgb = cv2.cvtColor(img_rgb[:, :, :3], cv2.COLOR_BGRA2BGR)
        elif len(img_rgb.shape) == 3:
            if img_rgb.shape[2] == 4:  # RGBA
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGRA2BGR)
    
    h, w = img_rgb.shape[:2]
    
    # Выводим информацию о размере изображения
    if debug:
        debug_print("\n" + "="*60)
        debug_print("📐 ИНФОРМАЦИЯ ОБ ИЗОБРАЖЕНИИ")
        debug_print("="*60)
        debug_print(f"Размер изображения: {w} x {h} пикселей")
        debug_print(f"Ориентация: {'Портретная' if h > w else 'Альбомная' if w > h else 'Квадратная'}")
        if isinstance(img, Image.Image):
            debug_print(f"Формат: {img.format if hasattr(img, 'format') else 'N/A'}")
            debug_print(f"Режим: {img.mode if hasattr(img, 'mode') else 'N/A'}")
        debug_print("="*60)
    
    # Инициализируем MediaPipe Face Detection и Face Mesh
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    result = {
        'detected': False,
        'face_count': 0,
        'landmarks': [],
        'bounding_boxes': [],
        'face_landmarks_2d': [],
        'face_landmarks_3d': [],
        'face_blendshapes': [],
        'face_geometry': {},
        'image_size': {'width': w, 'height': h}
    }
    
    # Face Detection
    with mp_face_detection.FaceDetection(
        model_selection=1,  # 0 для ближних лиц, 1 для дальних
        min_detection_confidence=0.5
    ) as face_detection:
        detection_results = face_detection.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        
        if debug:
            debug_print("\n" + "="*60)
            debug_print("🔍 MEDIAPIPE FACE DETECTION - ДЕБАГ ИНФОРМАЦИЯ")
            debug_print("="*60)
        
        if detection_results.detections:
            result['face_count'] = len(detection_results.detections)
            result['detected'] = True
            
            if debug:
                debug_print(f"✅ Обнаружено лиц: {result['face_count']}")
            
            for idx, detection in enumerate(detection_results.detections):
                # Bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                bounding_box = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'confidence': detection.score[0] if detection.score else 0.0
                }
                result['bounding_boxes'].append(bounding_box)
                
                if debug:
                    debug_print(f"\n--- Лицо #{idx + 1} ---")
                    debug_print(f"  Уверенность (confidence): {detection.score[0]:.4f}" if detection.score else "  Уверенность: N/A")
                    debug_print(f"  Bounding Box:")
                    debug_print(f"    X: {x}, Y: {y}")
                    debug_print(f"    Ширина: {width}, Высота: {height}")
                    debug_print(f"    Относительные координаты: x={bbox.xmin:.4f}, y={bbox.ymin:.4f}, w={bbox.width:.4f}, h={bbox.height:.4f}")
                
                # Key points (6 точек: глаза, нос, рот, уши)
                if detection.location_data.relative_keypoints:
                    if debug:
                        debug_print(f"  Ключевые точки (keypoints): {len(detection.location_data.relative_keypoints)}")
                    keypoints = []
                    for kp_idx, keypoint in enumerate(detection.location_data.relative_keypoints):
                        kp_x = int(keypoint.x * w)
                        kp_y = int(keypoint.y * h)
                        keypoints.append({
                            'x': kp_x,
                            'y': kp_y,
                            'relative_x': keypoint.x,
                            'relative_y': keypoint.y,
                            'name': ['right_eye', 'left_eye', 'nose_tip', 'mouth_center', 'right_ear', 'left_ear'][kp_idx] if kp_idx < 6 else f'point_{kp_idx}'
                        })
                        if debug:
                            debug_print(f"    {keypoints[-1]['name']}: ({kp_x}, {kp_y}) [относительно: ({keypoint.x:.4f}, {keypoint.y:.4f})]")
        else:
            if debug:
                debug_print("❌ Лица не обнаружены")
    
    # Face Mesh (468 landmarks)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,  # Включаем дополнительные landmarks (468 -> 468)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        mesh_results = face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        
        if debug:
            debug_print("\n" + "-"*60)
            debug_print("🔍 MEDIAPIPE FACE MESH - ДЕБАГ ИНФОРМАЦИЯ")
            debug_print("-"*60)
        
        if mesh_results.multi_face_landmarks:
            if debug:
                debug_print(f"✅ Обнаружено лиц в mesh: {len(mesh_results.multi_face_landmarks)}")
            
            for face_idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                landmarks_2d = []
                landmarks_3d = []
                
                if debug:
                    debug_print(f"\n--- Face Mesh #{face_idx + 1} ---")
                    debug_print(f"  Всего landmarks: {len(face_landmarks.landmark)}")
                
                for landmark_idx, landmark in enumerate(face_landmarks.landmark):
                    # 2D координаты (в пикселях)
                    x_2d = int(landmark.x * w)
                    y_2d = int(landmark.y * h)
                    z_2d = landmark.z * w  # z масштабируется по ширине
                    
                    landmarks_2d.append({
                        'x': x_2d,
                        'y': y_2d,
                        'z': z_2d,
                        'relative_x': landmark.x,
                        'relative_y': landmark.y,
                        'relative_z': landmark.z,
                        'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0,
                        'presence': landmark.presence if hasattr(landmark, 'presence') else 1.0
                    })
                    
                    landmarks_3d.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                result['face_landmarks_2d'].append(landmarks_2d)
                result['face_landmarks_3d'].append(landmarks_3d)
                result['landmarks'].append(landmarks_2d)
                
                if debug:
                    # Показываем некоторые важные landmarks
                    important_landmarks = {
                        10: 'Верхняя губа (верх)',
                        152: 'Подбородок',
                        33: 'Нос (кончик)',
                        468: 'Правый глаз (внешний угол)',
                        473: 'Левый глаз (внешний угол)',
                        0: 'Правый глаз (внутренний угол)',
                        227: 'Левый глаз (внутренний угол)',
                    }
                    debug_print(f"  Важные landmarks:")
                    for lm_idx, desc in important_landmarks.items():
                        if lm_idx < len(landmarks_2d):
                            lm = landmarks_2d[lm_idx]
                            debug_print(f"    [{lm_idx}] {desc}: ({lm['x']}, {lm['y']}) [z={lm['z']:.2f}, vis={lm['visibility']:.3f}]")
                    
                    # Статистика по z-координатам (глубина)
                    z_values = [lm['z'] for lm in landmarks_2d]
                    if z_values:
                        debug_print(f"  Глубина (z): min={min(z_values):.2f}, max={max(z_values):.2f}, mean={sum(z_values)/len(z_values):.2f}")
                    
                    # Статистика по visibility
                    vis_values = [lm['visibility'] for lm in landmarks_2d]
                    if vis_values:
                        debug_print(f"  Видимость: min={min(vis_values):.3f}, max={max(vis_values):.3f}, mean={sum(vis_values)/len(vis_values):.3f}")
        else:
            if debug:
                debug_print("❌ Face Mesh не обнаружил лица")
    
    if debug:
        debug_print("\n" + "="*60)
        debug_print("📊 ИТОГОВАЯ СВОДКА")
        debug_print("="*60)
        debug_print(f"Обнаружено лиц: {result['face_count']}")
        debug_print(f"Bounding boxes: {len(result['bounding_boxes'])}")
        debug_print(f"Face Mesh результатов: {len(result['face_landmarks_2d'])}")
        debug_print(f"Всего landmarks: {sum(len(lm) for lm in result['landmarks'])}")
        debug_print("="*60 + "\n")
    
    # Сохраняем дебаг информацию в файл, если указан путь
    if debug_file and debug_output:
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(debug_output))
            if debug:
                print(f"💾 Дебаг информация сохранена в файл: {debug_file}")
        except Exception as e:
            if debug:
                print(f"⚠️ Не удалось сохранить дебаг информацию в файл: {e}")
    
    return result if result['detected'] else None


def strip_background(portrait: Image.Image, model_name: str = "isnet-general-use") -> Image.Image:
    """
    Удаляет фон портрета с помощью rembg.
    
    Доступные модели:
    - 'isnet-general-use' - ISNet (лучшая точность, рекомендуется для портретов)
    - 'u2net_human_seg' - U2Net для людей (хорошо для портретов)
    - 'u2net' - U2Net базовая (быстрая, универсальная)
    - 'silueta' - Silueta (хорошая для общих случаев)
    - 'u2netp' - U2Net легкая версия (быстрая)
    """
    session = new_session(model_name)
    return remove(portrait, session=session)


def keep_largest_component(img: Image.Image) -> Image.Image:
    """
    Оставляет только самый большой связанный компонент в маске.
    Это помогает убрать людей на фоне, оставив только главного человека.
    """
    cv_img = pil_to_cv(img)
    alpha = cv_img[:, :, 3]
    
    # Бинаризуем маску (порог 127 для учета полупрозрачных пикселей)
    _, binary_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    
    # Находим все связанные компоненты
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    if num_labels <= 1:
        # Нет компонентов или только фон
        return img
    
    # Находим самый большой компонент (игнорируя фон с индексом 0)
    largest_component_idx = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    
    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_component_idx = i
    
    # Создаем маску только с самым большим компонентом
    largest_mask = (labels == largest_component_idx).astype(np.uint8) * 255
    
    # Применяем маску к альфа-каналу (сохраняем исходные значения альфа там, где есть компонент)
    alpha = np.where(largest_mask > 0, alpha, 0).astype(np.uint8)
    
    cv_img[:, :, 3] = alpha
    return cv_to_pil(cv_img)


def refine_alpha(
    img: Image.Image,
    erode: int = 0,
    dilate: int = 0,
    feather: int = 0,
    keep_largest: bool = True,
) -> Image.Image:
    """
    Улучшает маску: эрозия/дилятация и лёгкое размытие по краю.
    Все параметры — пиксели (целые, >=0). feather использует Gaussian blur.
    
    Args:
        keep_largest: Если True, оставляет только самый большой компонент (убирает людей на фоне)
    """
    cv_img = pil_to_cv(img)
    alpha = cv_img[:, :, 3]

    # Сначала оставляем только самый большой компонент
    if keep_largest:
        _, binary_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        if num_labels > 1:
            # Находим самый большой компонент (игнорируя фон с индексом 0)
            largest_component_idx = 1
            largest_area = stats[1, cv2.CC_STAT_AREA]
            
            for i in range(2, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > largest_area:
                    largest_area = area
                    largest_component_idx = i
            
            # Создаем маску только с самым большим компонентом
            largest_mask = (labels == largest_component_idx).astype(np.uint8) * 255
            # Применяем маску к альфа-каналу
            alpha = np.where(largest_mask > 0, alpha, 0).astype(np.uint8)

    if erode > 0:
        alpha = cv2.erode(alpha, np.ones((erode, erode), np.uint8), iterations=1)
    if dilate > 0:
        alpha = cv2.dilate(alpha, np.ones((dilate, dilate), np.uint8), iterations=1)
    if feather > 0:
        k = max(1, feather | 1)  # делаем ядро нечётным
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    alpha = np.clip(alpha, 0, 255).astype("uint8")
    cv_img[:, :, 3] = alpha
    return cv_to_pil(cv_img)


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL (RGBA) -> OpenCV (BGRA)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    """OpenCV (BGRA) -> PIL (RGBA)."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))


def center_face_horizontally(img: Image.Image, face_info: dict | None, debug: bool = False) -> Image.Image:
    """
    Центрирует лицо по горизонтали в изображении.
    Смещает изображение так, чтобы центр лица совпадал с центром изображения.
    
    Args:
        img: Изображение с альфа-каналом
        face_info: Информация о лице от MediaPipe (должен содержать bounding_boxes)
        debug: Если True, выводит информацию о центрировании
    
    Returns:
        Изображение с центрированным лицом (того же размера)
    """
    if not face_info or not face_info.get('bounding_boxes'):
        if debug:
            print("⚠️ Информация о лице отсутствует, пропускаем центрирование")
        return img
    
    # Берем первое (самое большое) лицо
    bbox = face_info['bounding_boxes'][0]
    face_center_x = bbox['x'] + bbox['width'] // 2
    img_width, img_height = img.size
    img_center_x = img_width // 2
    
    # Вычисляем смещение - на сколько нужно сдвинуть изображение
    offset_x = img_center_x - face_center_x
    
    if debug:
        print(f"\n🎯 ЦЕНТРИРОВАНИЕ ЛИЦА ПО ГОРИЗОНТАЛИ")
        print(f"  Bounding box: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}")
        print(f"  Центр лица: {face_center_x}px")
        print(f"  Центр изображения: {img_center_x}px")
        print(f"  Необходимое смещение: {offset_x}px")
    
    # Если лицо уже по центру (с небольшой погрешностью), возвращаем как есть
    if abs(offset_x) < 2:
        if debug:
            print("  ✅ Лицо уже по центру, смещение не требуется")
        return img
    
    # Создаем временное изображение с увеличенной шириной для смещения
    # Добавляем достаточно места с обеих сторон (минимум offset_x с каждой стороны)
    padding = abs(offset_x) + 100  # Добавляем запас
    temp_width = img_width + padding * 2
    temp_img = Image.new("RGBA", (temp_width, img_height), (0, 0, 0, 0))
    
    # Вычисляем позицию для вставки исходного изображения
    # Центр лица должен оказаться в центре временного изображения
    temp_center = temp_width // 2
    paste_x = temp_center - face_center_x
    
    # Вставляем исходное изображение
    temp_img.paste(img, (paste_x, 0), img)
    
    # Теперь обрезаем до исходного размера, центрируя лицо
    # Центр обрезки должен быть в центре временного изображения
    crop_start_x = temp_center - img_width // 2
    crop_end_x = crop_start_x + img_width
    result = temp_img.crop((crop_start_x, 0, crop_end_x, img_height))
    
    if debug:
        # Проверяем результат - распознаем лицо еще раз
        face_info_result = detect_face_mediapipe(result, debug=False)
        if face_info_result and face_info_result.get('bounding_boxes'):
            result_bbox = face_info_result['bounding_boxes'][0]
            result_face_center = result_bbox['x'] + result_bbox['width'] // 2
            result_img_center = result.size[0] // 2
            final_offset = result_img_center - result_face_center
            print(f"  ✅ Лицо центрировано")
            print(f"  Проверка: центр лица={result_face_center}px, центр изображения={result_img_center}px, отклонение={final_offset}px")
            if abs(final_offset) > 5:
                print(f"  ⚠️ ВНИМАНИЕ: Отклонение все еще велико!")
        else:
            print(f"  ✅ Изображение смещено (не удалось проверить результат)")
    
    return result


def expand_to_fill_width(img: Image.Image, padding: int = 0, min_scale: float = 1.12, debug: bool = False) -> Image.Image:
    """
    Масштабирует изображение так, чтобы видимая (непрозрачная) часть
    по ширине почти доходила до краёв кадра.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    img_width, img_height = img.size
    
    # Находим непрозрачные области через альфа-канал
    alpha = np.array(img.split()[3])
    coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
    if coords is None:
        if debug:
            print("⚠️ Не удалось найти непрозрачные области для масштабирования")
        return img

    x, y, w, h = cv2.boundingRect(coords)
    if w == 0:
        if debug:
            print("⚠️ Ширина непрозрачной области равна 0, пропускаем масштабирование")
        return img

    if debug:
        print(f"\n📏 АНАЛИЗ ДЛЯ МАСШТАБИРОВАНИЯ")
        print(f"  Размер изображения: {img_width} x {img_height}")
        print(f"  Bounding box непрозрачной области: x={x}, y={y}, w={w}, h={h}")
        print(f"  Текущая ширина объекта: {w}px из {img_width}px ({w/img_width*100:.1f}%)")

    # Целевая ширина (почти до краёв, небольшой отступ)
    target_width = img_width - 2 * padding

    # Масштаб: минимум min_scale, чтобы явно увеличить
    scale_needed = target_width / w
    scale = max(scale_needed, min_scale)
    
    if debug:
        print(f"  Целевая ширина: {target_width}px (padding={padding}px)")
        print(f"  Вычисленный масштаб: {scale_needed:.3f}, применяем: {scale:.3f}")

    # Масштабируем изображение
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    if debug:
        print(f"  Масштабируем: {img_width}x{img_height} -> {new_width}x{new_height}")
    
    resized = img.resize((new_width, new_height), Image.LANCZOS)

    # Центрируем обратно в исходный размер
    result = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    offset_x = (img_width - new_width) // 2
    offset_y = (img_height - new_height) // 2
    result.paste(resized, (offset_x, offset_y), resized)

    if debug:
        # Проверяем результат
        result_alpha = np.array(result.split()[3])
        result_coords = cv2.findNonZero((result_alpha > 0).astype(np.uint8))
        if result_coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(result_coords)
            print(f"  ✅ После масштабирования: ширина объекта={rw}px из {img_width}px ({rw/img_width*100:.1f}%)")
            print(f"  Offset: ({offset_x}, {offset_y})")

    return result


def align_eyes_vertical(
    img: Image.Image,
    face_info: dict | None,
    target_frac: float = 1 / 3,
    debug: bool = False,
) -> Image.Image:
    """
    Смещает изображение по вертикали так, чтобы линия глаз была на заданной доле высоты.
    target_frac=1/3 означает линия глаз на 1/3 от высоты сверху.
    """
    if not face_info:
        if debug:
            print("⚠️ Нет информации о лице для вертикального выравнивания")
        return img

    img_w, img_h = img.size

    # Берём первый face_mesh landmarks, если есть
    eye_y = None
    if face_info.get("face_landmarks_2d"):
        lms = face_info["face_landmarks_2d"][0]
        idxs = [33, 263]  # внешние уголки глаз
        vals = [lms[i]["y"] for i in idxs if i < len(lms)]
        if vals:
            eye_y = sum(vals) / len(vals)

    # Фолбек: по bounding box
    if eye_y is None and face_info.get("bounding_boxes"):
        bbox = face_info["bounding_boxes"][0]
        eye_y = bbox["y"] + bbox["height"] * 0.35

    if eye_y is None:
        if debug:
            print("⚠️ Не удалось вычислить линию глаз, пропускаем вертикальное выравнивание")
        return img

    target_y = img_h * target_frac
    offset_y = target_y - eye_y

    if debug:
        print("\n📐 ВЫРАВНИВАНИЕ ПО ВЕРТИКАЛИ (глаза на 1/3)")
        print(f"  Eye Y: {eye_y:.1f}px, Target Y: {target_y:.1f}px")
        print(f"  Смещение: {offset_y:.1f}px")

    if abs(offset_y) < 1:
        if debug:
            print("  ✅ Смещение не требуется")
        return img

    # Создаем временное изображение с запасом по высоте
    padding = int(abs(offset_y)) + 100
    temp_h = img_h + padding * 2
    temp_w = img_w
    temp = Image.new("RGBA", (temp_w, temp_h), (0, 0, 0, 0))

    # Вставляем исходное изображение в позицию, где глаза будут на target_y
    # target_y в координатах временного изображения
    temp_target_y = target_y + padding  # Добавляем padding сверху
    paste_y = int(temp_target_y - eye_y)
    temp.paste(img, (0, paste_y), img)

    # Обрезаем до исходной высоты, сохраняя глаза на target_y
    # Обрезаем так, чтобы target_y остался на той же позиции
    crop_start_y = padding
    crop_end_y = crop_start_y + img_h
    result = temp.crop((0, crop_start_y, img_w, crop_end_y))

    if debug:
        # Проверка после сдвига
        face_info_check = detect_face_mediapipe(result, debug=False)
        if face_info_check and face_info_check.get("face_landmarks_2d"):
            lms2 = face_info_check["face_landmarks_2d"][0]
            vals2 = [lms2[i]["y"] for i in [33, 263] if i < len(lms2)]
            if vals2:
                new_eye_y = sum(vals2) / len(vals2)
                print(f"  ✅ Проверка: eye_y={new_eye_y:.1f}px, цель={target_y:.1f}px, отклонение={(target_y - new_eye_y):.1f}px")

    return result


def fill_bottom_gap_with_last_pixels(img: Image.Image, debug: bool = False) -> Image.Image:
    """
    Растягивает нижние пиксели до низа, если снизу есть пустота (прозрачность).
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = np.array(img.split()[3])
    coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    img_w, img_h = img.size
    bottom = y + h
    gap = img_h - bottom
    if gap <= 0:
        return img

    # Берем нижнюю полоску и растягиваем
    strip_h = min(4, h) if h > 0 else 1
    strip_top = max(bottom - strip_h, 0)
    strip = img.crop((0, strip_top, img_w, bottom))
    stretched = strip.resize((img_w, gap + strip_h), Image.BILINEAR)

    result = img.copy()
    paste_y = strip_top
    result.paste(stretched, (0, paste_y), stretched)

    if debug:
        print(f"🔻 Заполнение низа: gap={gap}px, strip_h={strip_h}px, paste_y={paste_y}")

    return result


def fit_to_size(
    img: Image.Image,
    target_size: tuple[int, int],
    anchor_y: float = 0.5,
) -> Image.Image:
    """
    Вписывает изображение в целевой размер с сохранением пропорций.
    Добавляет прозрачный фон, если изображение меньше целевого размера.
    anchor_y — вертикальная привязка (0=сверху, 0.5=центр, 1=снизу).
    """
    target_width, target_height = target_size
    img_width, img_height = img.size
    
    scale = min(target_width / img_width, target_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    resized = img.resize((new_width, new_height), Image.LANCZOS)
    
    result = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
    
    x_offset = (target_width - new_width) // 2
    y_space = target_height - new_height
    y_offset = int(y_space * anchor_y)
    y_offset = max(0, min(y_offset, target_height - new_height))
    
    result.paste(resized, (x_offset, y_offset), resized)
    
    return result


def reinhard_color_transfer(
    source_bgr: np.ndarray, target_bgr: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Перенос цвета по методике Рейнхарда (Lab).
    Если передана маска (uint8 0/255), статистика считается по ней,
    а преобразование применяется только к замаскированной области.
    """
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype("float32")

    if mask is None:
        mask = np.full(source_lab.shape[:2], 255, dtype=np.uint8)

    # Средние и сигмы по маске
    (l_mean_src, a_mean_src, b_mean_src), (l_std_src, a_std_src, b_std_src) = (
        cv2.meanStdDev(source_lab, mask=mask)
    )
    (l_mean_tgt, a_mean_tgt, b_mean_tgt), (l_std_tgt, a_std_tgt, b_std_tgt) = (
        cv2.meanStdDev(target_lab)
    )

    # Предотвращаем деление на ноль
    l_std_src = l_std_src + 1e-6
    a_std_src = a_std_src + 1e-6
    b_std_src = b_std_src + 1e-6

    result_lab = source_lab.copy()
    m = mask > 0

    # Применяем преобразование только внутри маски
    for channel_idx, (mean_src, std_src, mean_tgt, std_tgt) in enumerate(
        [
            (l_mean_src, l_std_src, l_mean_tgt, l_std_tgt),
            (a_mean_src, a_std_src, a_mean_tgt, a_std_tgt),
            (b_mean_src, b_std_src, b_mean_tgt, b_std_tgt),
        ]
    ):
        channel = result_lab[:, :, channel_idx]
        channel[m] = (
            (channel[m] - mean_src[0]) * (std_tgt[0] / std_src[0]) + mean_tgt[0]
        )

    result_lab = np.clip(result_lab, 0, 255).astype("uint8")
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def apply_color_reference(
    portrait_rgba: Image.Image,
    ref_image: Image.Image,
    color_strength: float = 1.0,
    reduce_contrast: float = 0.85,
    brightness_adjust: float = 0.0,
    saturation_adjust: float = 0.0,
) -> Image.Image:
    """
    Приводит портрет к цветам референса, сохраняя альфа-канал.
    
    Args:
        portrait_rgba: Портрет с альфа-каналом
        ref_image: Референсное изображение
        color_strength: Интенсивность переноса цвета (0.0-1.0, где 1.0 = полный перенос, 0.0 = без изменений)
        reduce_contrast: Коэффициент снижения контраста (0.0-1.0, где 1.0 = без изменений, 0.85 = снижение на 15%)
        brightness_adjust: Коррекция яркости (-1.0 до 1.0, где 0.0 = без изменений, положительные = ярче)
        saturation_adjust: Коррекция насыщенности (-1.0 до 1.0, где 0.0 = без изменений, положительные = насыщеннее)
    """
    portrait_cv = pil_to_cv(portrait_rgba)
    ref_cv = pil_to_cv(ref_image)

    alpha = portrait_cv[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255

    # Цвета только из непрозрачной части
    source_bgr = portrait_cv[:, :, :3]
    target_bgr = ref_cv[:, :, :3]

    if mask.sum() == 0:
        return portrait_rgba

    # Применяем перенос цвета с учетом интенсивности
    if color_strength > 0:
        transferred_bgr = reinhard_color_transfer(source_bgr, target_bgr, mask=mask)
        if color_strength < 1.0:
            # Смешиваем исходное и переданное изображение
            transferred_bgr = cv2.addWeighted(
                source_bgr, 1.0 - color_strength, transferred_bgr, color_strength, 0
            )
    else:
        transferred_bgr = source_bgr.copy()
    
    # Конвертируем в LAB для работы с яркостью и насыщенностью
    lab = cv2.cvtColor(transferred_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Снижаем контраст для более мягкого результата
    if reduce_contrast < 1.0:
        l_channel = lab[:, :, 0]
        mean_l = l_channel[mask > 0].mean() if mask.sum() > 0 else l_channel.mean()
        l_channel = l_channel * reduce_contrast + mean_l * (1 - reduce_contrast)
        lab[:, :, 0] = np.clip(l_channel, 0, 255)
    
    # Коррекция яркости
    if brightness_adjust != 0.0:
        l_channel = lab[:, :, 0]
        adjustment = brightness_adjust * 50  # Масштабируем до разумного диапазона
        l_channel = np.clip(l_channel + adjustment, 0, 255)
        lab[:, :, 0] = l_channel
    
    # Конвертируем обратно в BGR
    transferred_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Коррекция насыщенности (работаем в HSV)
    if saturation_adjust != 0.0:
        hsv = cv2.cvtColor(transferred_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        s_channel = hsv[:, :, 1]
        adjustment = saturation_adjust * 100  # Масштабируем до разумного диапазона
        s_channel = np.clip(s_channel + adjustment, 0, 255)
        hsv[:, :, 1] = s_channel
        transferred_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    portrait_cv[:, :, :3] = transferred_bgr
    return cv_to_pil(portrait_cv)


def apply_sepia(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Накладывает сепию (коричневатый оттенок) на изображение.
    
    Args:
        img: Изображение с альфа-каналом
        strength: Интенсивность эффекта сепии (0.0-1.0, где 1.0 = полный эффект)
    
    Returns:
        Изображение с эффектом сепии
    """
    if strength <= 0:
        return img
    
    # Конвертируем в RGB для обработки
    if img.mode == "RGBA":
        rgb_img = img.convert("RGB")
        has_alpha = True
        alpha = img.split()[3]
    else:
        rgb_img = img.convert("RGB")
        has_alpha = False
    
    # Матрица сепии (классическая)
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    # Применяем матрицу сепии
    img_array = np.array(rgb_img).astype(np.float32)
    sepia_array = img_array @ sepia_matrix.T
    sepia_array = np.clip(sepia_array, 0, 255).astype(np.uint8)
    
    # Смешиваем с исходным изображением в зависимости от strength
    if strength < 1.0:
        sepia_array = (img_array * (1 - strength) + sepia_array * strength).astype(np.uint8)
    
    result = Image.fromarray(sepia_array, mode="RGB")
    
    # Восстанавливаем альфа-канал, если был
    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    
    return result


def enhance_face_gfpgan(
    img_rgba: Image.Image, upscale: int = 1, strength: float = 1.0, iterations: int = 1
) -> Image.Image:
    """
    Сглаживание/улучшение лица через GFPGAN (если установлен).
    
    Args:
        img_rgba: Входное изображение с альфа-каналом
        upscale: Масштаб увеличения для обработки (1-4, рекомендуется 2-4)
        strength: Интенсивность эффекта (0.0-1.0, где 1.0 = полный эффект)
        iterations: Количество итераций улучшения (1-3, больше = сильнее эффект)
    """
    # Шим: в новых torchvision модуль functional_tensor отсутствует, basicsr его ждет.
    try:
        import torchvision.transforms.functional_tensor as _  # type: ignore
    except ImportError:
        from torchvision.transforms import functional as F

        ft_module = types.ModuleType("torchvision.transforms.functional_tensor")
        ft_module.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = ft_module

    try:
        from gfpgan import GFPGANer
    except ImportError as exc:  # pragma: no cover - внешняя зависимость
        raise RuntimeError(
            "GFPGAN не установлен. Установите: pip install gfpgan"
        ) from exc

    cv_img = pil_to_cv(img_rgba)
    alpha = cv_img[:, :, 3]
    bgr = cv_img[:, :, :3]
    original_bgr = bgr.copy()

    restorer = GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

    # Применяем улучшение несколько раз для более сильного эффекта
    restored_bgr = bgr.copy()
    for _ in range(iterations):
        # Возвращает: cropped_faces, restored_faces, restored_img
        _, _, restored_bgr = restorer.enhance(
            restored_bgr, has_aligned=False, only_center_face=False, paste_back=True
        )
        
        # GFPGAN может менять размер (upscale>1). Возвращаем к исходному.
        if restored_bgr.shape[:2] != original_bgr.shape[:2]:
            restored_bgr = cv2.resize(
                restored_bgr,
                (original_bgr.shape[1], original_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

    # Смешиваем исходное и улучшенное изображение в зависимости от strength
    if strength < 1.0:
        restored_bgr = cv2.addWeighted(
            original_bgr, 1.0 - strength, restored_bgr, strength, 0
        )

    cv_img[:, :, :3] = restored_bgr
    cv_img[:, :, 3] = alpha
    return cv_to_pil(cv_img)


def process(
    portrait_path: str,
    ref_path: str,
    output_path: str,
    use_face_enhance: bool = True,
    face_upscale: int = 4,
    face_strength: float = 0.7,
    face_iterations: int = 2,
    background_path: str | None = None,
    alpha_erode: int = 17,
    alpha_dilate: int = 0,
    alpha_feather: int = 16,
    bg_model: str = "u2net_human_seg",
    keep_largest: bool = True,
    face_detect: bool = True,
    center_face: bool = False,
    normalize_exposure: bool = False,
    color_strength: float = 1.0,
    reduce_contrast: float = 0.85,
    brightness_adjust: float = 0.0,
    saturation_adjust: float = 0.0,
    sepia_strength: float = 0.0,
) -> float:
    """
    Обрабатывает портрет и возвращает время обработки в секундах.
    """
    start_time = time.time()
    
    portrait = load_image(portrait_path, debug_orientation=face_detect)
    # Нормализуем экспозицию входного изображения (если включено)
    if normalize_exposure:
        portrait = normalize_exposure(portrait, debug=face_detect)
    ref = load_image(ref_path)

    # Распознавание лица с помощью MediaPipe (для дебага и центрирования)
    face_info = None
    if face_detect:
        print("\n" + "🔍" * 30)
        print("НАЧАЛО РАСПОЗНАВАНИЯ ЛИЦА С MEDIAPIPE")
        print("🔍" * 30)
        # Сохраняем дебаг информацию в файл рядом с результатом
        debug_file_path = output_path.replace('.png', '_mediapipe_debug.txt').replace('.jpg', '_mediapipe_debug.txt')
        face_info = detect_face_mediapipe(portrait, debug=True, debug_file=debug_file_path)
        if face_info:
            print("✅ Распознавание лица завершено успешно")
            print(f"💾 Дебаг информация сохранена в: {debug_file_path}")
        else:
            print("⚠️ Лицо не обнаружено или MediaPipe недоступен")
        print("🔍" * 30 + "\n")

    portrait_no_bg = strip_background(portrait, model_name=bg_model)
    portrait_no_bg = refine_alpha(
        portrait_no_bg,
        erode=alpha_erode,
        dilate=alpha_dilate,
        feather=alpha_feather,
        keep_largest=keep_largest,  # Оставляем только главного человека (по умолчанию True)
    )
    
    # Центрируем лицо по горизонтали (только если включено)
    if center_face:
        # Нужно пересчитать координаты лица для изображения без фона
        face_info_no_bg = detect_face_mediapipe(portrait_no_bg, debug=False)
        if face_info_no_bg:
            if face_detect:
                print("\n🎯 ПРИМЕНЕНИЕ ЦЕНТРИРОВАНИЯ ЛИЦА")
            portrait_no_bg = center_face_horizontally(portrait_no_bg, face_info_no_bg, debug=face_detect)
            # После сдвига масштабируем, чтобы непрозрачная область доходила до краёв
            portrait_no_bg = expand_to_fill_width(portrait_no_bg, padding=0, min_scale=1.12, debug=face_detect)
            # Обновляем инфо о лице после масштабирования
            face_info_scaled = detect_face_mediapipe(portrait_no_bg, debug=False)
            # Выравниваем по вертикали: линия глаз на 1/3 высоты
            portrait_no_bg = align_eyes_vertical(portrait_no_bg, face_info_scaled or face_info_no_bg, target_frac=1/3, debug=face_detect)
            # Заполняем низ растяжением нижних пикселей, чтобы убрать пустоту
            portrait_no_bg = fill_bottom_gap_with_last_pixels(portrait_no_bg, debug=face_detect)
            if face_detect:
                print("🎯" * 30 + "\n")
    elif face_detect:
        print("\n⚠️ Лицо не обнаружено на изображении без фона, центрирование пропущено\n")
    colored = apply_color_reference(
        portrait_no_bg,
        ref,
        color_strength=color_strength,
        reduce_contrast=reduce_contrast,
        brightness_adjust=brightness_adjust,
        saturation_adjust=saturation_adjust,
    )
    if use_face_enhance:
        colored = enhance_face_gfpgan(
            colored,
            upscale=face_upscale,
            strength=face_strength,
            iterations=face_iterations,
        )

    if background_path and os.path.exists(background_path):
        bg = Image.open(background_path).convert("RGBA")
        bg_resized = bg.resize(colored.size, Image.LANCZOS)
        # Кладём портрет сверху (с альфой) на фон
        colored = Image.alpha_composite(bg_resized, colored)

    # Вписываем изображение в размер 720x1280 с сохранением пропорций
    # anchor_y=1/3 чтобы линия глаз осталась ближе к верхней трети
    colored = fit_to_size(colored, (720, 1280), anchor_y=1/3)
    
    # Накладываем сепию (если включено)
    if sepia_strength > 0:
        colored = apply_sepia(colored, strength=sepia_strength)

    colored.save(output_path)
    
    elapsed_time = time.time() - start_time
    return elapsed_time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Удаление фона портрета и приведение к цветам референса."
    )
    parser.add_argument("portrait", help="Путь к исходному портрету")
    parser.add_argument(
        "reference",
        nargs="?",
        default="src/ref.png",
        help="Путь к изображению-референсу по цвету (по умолчанию: src/ref.png)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="result.png",
        help="Путь для сохранения результата (PNG с прозрачным фоном)",
    )
    parser.add_argument(
        "--face-enhance",
        action="store_true",
        help="Сгладить/улучшить лицо через GFPGAN (требуется pip install gfpgan). По умолчанию включено. Используйте --no-face-enhance для отключения.",
    )
    parser.add_argument(
        "--no-face-enhance",
        action="store_false",
        dest="face_enhance",
        help="Отключить улучшение лица",
    )
    parser.add_argument(
        "--face-upscale",
        type=int,
        default=4,
        help="Масштаб увеличения для GFPGAN (1-4, рекомендуется 2-4, выше = лучше качество обработки)",
    )
    parser.add_argument(
        "--face-strength",
        type=float,
        default=0.7,
        help="Интенсивность улучшения лица (0.0-1.0, где 1.0 = полный эффект, по умолчанию 0.7)",
    )
    parser.add_argument(
        "--face-iterations",
        type=int,
        default=2,
        help="Количество итераций улучшения (1-3, больше = сильнее эффект, по умолчанию 2)",
    )
    parser.add_argument(
        "--background",
        default="src/bg.jpg",
        help="Путь к фону (RGBA/RGB). Будет подложен под итоговое изображение (по умолчанию: src/bg.jpg). Используйте --no-background для отключения.",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Не использовать фон (игнорирует --background)",
    )
    parser.add_argument(
        "--alpha-erode",
        type=int,
        default=17,
        help="Сжать маску на N пикселей (эрозия) для уборки ореолов",
    )
    parser.add_argument(
        "--alpha-dilate",
        type=int,
        default=0,
        help="Расширить маску на N пикселей (дилятация) после эрозии",
    )
    parser.add_argument(
        "--alpha-feather",
        type=int,
        default=16,
        help="Размытие края маски (Gaussian blur, пиксели) для мягкого перехода",
    )
    parser.add_argument(
        "--bg-model",
        type=str,
        default="u2net_human_seg",
        choices=["isnet-general-use", "u2net_human_seg", "u2net", "silueta", "u2netp"],
        help="Модель для удаления фона: u2net_human_seg (для людей, по умолчанию), isnet-general-use (лучшая), u2net (базовая), silueta, u2netp (легкая)",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Оставить все объекты на маске (не только главного человека). По умолчанию оставляется только самый большой объект.",
    )
    parser.add_argument(
        "--preset",
        choices=["face3", "face8"],
        help="Предустановки: face3 (upscale=3, фон bg.jpg), face8 (upscale=8, фон bg.jpg)",
    )
    parser.add_argument(
        "--face-detect",
        action="store_true",
        default=True,
        help="Включить распознавание лица с помощью MediaPipe для дебага (по умолчанию включено)",
    )
    parser.add_argument(
        "--no-face-detect",
        action="store_false",
        dest="face_detect",
        help="Отключить распознавание лица с помощью MediaPipe",
    )
    parser.add_argument(
        "--center-face",
        action="store_true",
        default=False,
        help="Включить центрирование лица по горизонтали и вертикали (по умолчанию отключено)",
    )
    parser.add_argument(
        "--normalize-exposure",
        action="store_true",
        default=False,
        help="Включить нормализацию экспозиции (увеличение яркости) входного изображения (по умолчанию отключено)",
    )
    parser.add_argument(
        "--color-strength",
        type=float,
        default=1.0,
        help="Интенсивность переноса цвета из референса (0.0-1.0, по умолчанию 1.0 = полный перенос)",
    )
    parser.add_argument(
        "--reduce-contrast",
        type=float,
        default=0.85,
        help="Коэффициент снижения контраста после переноса цвета (0.0-1.0, по умолчанию 0.85)",
    )
    parser.add_argument(
        "--brightness-adjust",
        type=float,
        default=0.0,
        help="Коррекция яркости (-1.0 до 1.0, по умолчанию 0.0 = без изменений)",
    )
    parser.add_argument(
        "--saturation-adjust",
        type=float,
        default=0.0,
        help="Коррекция насыщенности (-1.0 до 1.0, по умолчанию 0.0 = без изменений)",
    )
    parser.add_argument(
        "--sepia",
        type=float,
        default=0.0,
        help="Интенсивность эффекта сепии (0.0-1.0, по умолчанию 0.0 = отключено)",
    )
    args = parser.parse_args()

    # Устанавливаем значения по умолчанию
    if not hasattr(args, 'face_enhance') or args.face_enhance is None:
        args.face_enhance = True
    
    # Проверяем наличие файлов по умолчанию
    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Референс не найден: {args.reference}")
    
    # Обрабатываем фон
    if args.no_background:
        args.background = None
    elif args.background == "src/bg.jpg" and not os.path.exists("src/bg.jpg"):
        # Если файл по умолчанию не существует, не используем фон
        args.background = None
    elif args.background and not os.path.exists(args.background):
        raise FileNotFoundError(f"Фон не найден: {args.background}")

    # Пресеты: переопределяют ключевые опции, можно дополнительно менять руками.
    if args.preset:
        # Проверяем наличие bg.jpg в src/ или в корне
        bg_path = "src/bg.jpg" if os.path.exists("src/bg.jpg") else "bg.jpg"
        preset_map = {
            "face3": {
                "face_enhance": True,
                "face_upscale": 3,
                "face_strength": 1.0,
                "face_iterations": 1,
                "background": bg_path,
            },
            "face8": {
                "face_enhance": True,
                "face_upscale": 4,
                "face_strength": 1.0,
                "face_iterations": 2,
                "background": bg_path,
            },
        }
        preset = preset_map[args.preset]
        args.face_enhance = preset["face_enhance"]
        args.face_upscale = preset["face_upscale"]
        args.face_strength = preset["face_strength"]
        args.face_iterations = preset["face_iterations"]
        if args.background is None:
            args.background = preset["background"]

    process(
        args.portrait,
        args.reference,
        args.output,
        use_face_enhance=args.face_enhance,
        face_upscale=args.face_upscale,
        face_strength=args.face_strength,
        face_iterations=args.face_iterations,
        background_path=args.background,
        alpha_erode=args.alpha_erode,
        alpha_dilate=args.alpha_dilate,
        alpha_feather=args.alpha_feather,
        bg_model=args.bg_model,
        keep_largest=not args.keep_all,  # Если --keep-all, то keep_largest=False
        face_detect=args.face_detect,
        center_face=args.center_face,
        normalize_exposure=args.normalize_exposure,
        color_strength=args.color_strength,
        reduce_contrast=args.reduce_contrast,
        brightness_adjust=args.brightness_adjust,
        saturation_adjust=args.saturation_adjust,
        sepia_strength=args.sepia,
    )


if __name__ == "__main__":
    main()

