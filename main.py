import argparse
import cv2
import numpy as np
from rembg import remove
from PIL import Image


def load_image(path: str) -> Image.Image:
    """Загружает изображение в формате RGBA."""
    img = Image.open(path).convert("RGBA")
    return img


def strip_background(portrait: Image.Image) -> Image.Image:
    """Удаляет фон портрета с помощью rembg."""
    return remove(portrait)


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL (RGBA) -> OpenCV (BGRA)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    """OpenCV (BGRA) -> PIL (RGBA)."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))


def reinhard_color_transfer(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Перенос цвета по методике Рейнхарда (работает в пространстве Lab)."""
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype("float32")

    (l_mean_src, a_mean_src, b_mean_src), (l_std_src, a_std_src, b_std_src) = (
        cv2.meanStdDev(source_lab)
    )
    (l_mean_tgt, a_mean_tgt, b_mean_tgt), (l_std_tgt, a_std_tgt, b_std_tgt) = (
        cv2.meanStdDev(target_lab)
    )

    l, a, b = cv2.split(source_lab)
    l = (l - l_mean_src[0]) * (l_std_tgt[0] / (l_std_src[0] + 1e-6)) + l_mean_tgt[0]
    a = (a - a_mean_src[0]) * (a_std_tgt[0] / (a_std_src[0] + 1e-6)) + a_mean_tgt[0]
    b = (b - b_mean_src[0]) * (b_std_tgt[0] / (b_std_src[0] + 1e-6)) + b_mean_tgt[0]

    transfer_lab = cv2.merge([l, a, b])
    transfer_lab = np.clip(transfer_lab, 0, 255).astype("uint8")
    transfer_bgr = cv2.cvtColor(transfer_lab, cv2.COLOR_LAB2BGR)
    return transfer_bgr


def apply_color_reference(portrait_rgba: Image.Image, ref_image: Image.Image) -> Image.Image:
    """Приводит портрет к цветам референса, сохраняя альфа-канал."""
    portrait_cv = pil_to_cv(portrait_rgba)
    ref_cv = pil_to_cv(ref_image)

    # Отдельно передаем цвет только по непрозрачной области
    alpha = portrait_cv[:, :, 3]
    color_src = cv2.cvtColor(portrait_cv[:, :, :3], cv2.COLOR_BGR2RGB)

    # Заполняем фон средним цветом референса, чтобы избежать артефактов на прозрачности
    ref_mean = cv2.mean(ref_cv[:, :, :3])[:3]
    bg = np.full_like(color_src, ref_mean)
    composited = color_src.copy()
    mask = alpha > 0
    composited[~mask] = bg[~mask]

    transferred = reinhard_color_transfer(composited[:, :, ::-1], ref_cv[:, :, :3])
    portrait_cv[:, :, :3] = transferred[:, :, ::-1]
    return cv_to_pil(portrait_cv)


def process(portrait_path: str, ref_path: str, output_path: str) -> None:
    portrait = load_image(portrait_path)
    ref = load_image(ref_path)

    portrait_no_bg = strip_background(portrait)
    colored = apply_color_reference(portrait_no_bg, ref)
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
    args = parser.parse_args()

    process(args.portrait, args.reference, args.output)


if __name__ == "__main__":
    main()

