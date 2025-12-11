import argparse
import os
import sys
import types

import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image


def load_image(path: str) -> Image.Image:
    """Загружает изображение в формате RGBA."""
    img = Image.open(path).convert("RGBA")
    return img


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


def refine_alpha(
    img: Image.Image, erode: int = 0, dilate: int = 0, feather: int = 0
) -> Image.Image:
    """
    Улучшает маску: эрозия/дилятация и лёгкое размытие по краю.
    Все параметры — пиксели (целые, >=0). feather использует Gaussian blur.
    """
    cv_img = pil_to_cv(img)
    alpha = cv_img[:, :, 3]

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


def apply_color_reference(portrait_rgba: Image.Image, ref_image: Image.Image) -> Image.Image:
    """Приводит портрет к цветам референса, сохраняя альфа-канал."""
    portrait_cv = pil_to_cv(portrait_rgba)
    ref_cv = pil_to_cv(ref_image)

    alpha = portrait_cv[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255

    # Цвета только из непрозрачной части
    source_bgr = portrait_cv[:, :, :3]
    target_bgr = ref_cv[:, :, :3]

    if mask.sum() == 0:
        return portrait_rgba

    transferred_bgr = reinhard_color_transfer(source_bgr, target_bgr, mask=mask)
    portrait_cv[:, :, :3] = transferred_bgr
    return cv_to_pil(portrait_cv)


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
    use_face_enhance: bool = False,
    face_upscale: int = 1,
    face_strength: float = 1.0,
    face_iterations: int = 1,
    background_path: str | None = None,
    alpha_erode: int = 0,
    alpha_dilate: int = 0,
    alpha_feather: int = 0,
    bg_model: str = "isnet-general-use",
) -> None:
    portrait = load_image(portrait_path)
    ref = load_image(ref_path)

    portrait_no_bg = strip_background(portrait, model_name=bg_model)
    portrait_no_bg = refine_alpha(
        portrait_no_bg, erode=alpha_erode, dilate=alpha_dilate, feather=alpha_feather
    )
    colored = apply_color_reference(portrait_no_bg, ref)
    if use_face_enhance:
        colored = enhance_face_gfpgan(
            colored,
            upscale=face_upscale,
            strength=face_strength,
            iterations=face_iterations,
        )

    if background_path:
        bg = Image.open(background_path).convert("RGBA")
        bg_resized = bg.resize(colored.size, Image.LANCZOS)
        # Кладём портрет сверху (с альфой) на фон
        colored = Image.alpha_composite(bg_resized, colored)

    colored.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Удаление фона портрета и приведение к цветам референса."
    )
    parser.add_argument("portrait", help="Путь к исходному портрету")
    parser.add_argument("reference", help="Путь к изображению-референсу по цвету")
    parser.add_argument(
        "-o",
        "--output",
        default="result.png",
        help="Путь для сохранения результата (PNG с прозрачным фоном)",
    )
    parser.add_argument(
        "--face-enhance",
        action="store_true",
        help="Сгладить/улучшить лицо через GFPGAN (требуется pip install gfpgan)",
    )
    parser.add_argument(
        "--face-upscale",
        type=int,
        default=2,
        help="Масштаб увеличения для GFPGAN (1-4, рекомендуется 2-4, выше = лучше качество обработки)",
    )
    parser.add_argument(
        "--face-strength",
        type=float,
        default=1.0,
        help="Интенсивность улучшения лица (0.0-1.0, где 1.0 = полный эффект, по умолчанию 1.0)",
    )
    parser.add_argument(
        "--face-iterations",
        type=int,
        default=1,
        help="Количество итераций улучшения (1-3, больше = сильнее эффект, по умолчанию 1)",
    )
    parser.add_argument(
        "--background",
        help="Путь к фону (RGBA/RGB). Будет подложен под итоговое изображение.",
    )
    parser.add_argument(
        "--alpha-erode",
        type=int,
        default=0,
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
        default=0,
        help="Размытие края маски (Gaussian blur, пиксели) для мягкого перехода",
    )
    parser.add_argument(
        "--bg-model",
        type=str,
        default="isnet-general-use",
        choices=["isnet-general-use", "u2net_human_seg", "u2net", "silueta", "u2netp"],
        help="Модель для удаления фона: isnet-general-use (лучшая, по умолчанию), u2net_human_seg (для людей), u2net (базовая), silueta, u2netp (легкая)",
    )
    parser.add_argument(
        "--preset",
        choices=["face3", "face8"],
        help="Предустановки: face3 (upscale=3, фон bg.jpg), face8 (upscale=8, фон bg.jpg)",
    )
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()

