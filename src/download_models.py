"""
Скрипт для предзагрузки всех моделей удаления фона.
Запустите этот скрипт, чтобы скачать все модели заранее.
"""

from rembg import remove, new_session
from PIL import Image
import sys


def download_models():
    """Предзагружает все доступные модели."""
    models = [
        ("isnet-general-use", "ISNet (лучшая точность)"),
        ("u2net_human_seg", "U2Net для людей"),
        ("u2net", "U2Net базовая"),
        ("silueta", "Silueta"),
        ("u2netp", "U2Net Lite"),
    ]

    # Создаем тестовое изображение для инициализации модели
    test_img = Image.new("RGB", (100, 100), color="red")

    print("=" * 60)
    print("Предзагрузка моделей удаления фона")
    print("=" * 60)
    print()

    success_count = 0
    failed_models = []

    for model_name, description in models:
        print(f"Загрузка: {model_name} ({description})...", end=" ", flush=True)
        try:
            # Создаем сессию модели (это скачает модель, если её нет)
            session = new_session(model_name)
            # Тестируем на маленьком изображении
            result = remove(test_img, session=session)
            print("✓ Успешно")
            success_count += 1
        except Exception as e:
            print(f"✗ Ошибка: {e}")
            failed_models.append((model_name, str(e)))

    print()
    print("=" * 60)
    print(f"Загружено моделей: {success_count}/{len(models)}")
    if failed_models:
        print("\nОшибки:")
        for model_name, error in failed_models:
            print(f"  - {model_name}: {error}")
    print("=" * 60)

    if success_count == len(models):
        print("\n✓ Все модели успешно загружены!")
        return 0
    else:
        print(f"\n⚠ Загружено {success_count} из {len(models)} моделей")
        return 1


if __name__ == "__main__":
    sys.exit(download_models())

