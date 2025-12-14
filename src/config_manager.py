"""
Менеджер конфигурации для сохранения и загрузки настроек по умолчанию.
"""
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any


# Используем абсолютный путь относительно корня проекта
CONFIG_FILE = Path(__file__).parent.parent / "config.json"


def get_default_config() -> Dict[str, Any]:
    """Возвращает настройки по умолчанию."""
    return {
        "face_enhance": True,
        "face_upscale": 4,
        "face_strength": 0.7,
        "face_iterations": 2,
        "center_face": False,
        "normalize_exposure": False,
        "color_strength": 1.0,
        "reduce_contrast": 0.85,
        "brightness_adjust": 0.0,
        "saturation_adjust": 0.0,
        "sepia_strength": 0.0,
        "keep_largest": True,
        "alpha_erode": 17,
        "alpha_dilate": 0,
        "alpha_feather": 16,
        "bg_model": "u2net_human_seg",
        "face_detect": True,
    }


def load_config() -> Dict[str, Any]:
    """
    Загружает настройки из конфигурационного файла.
    Если файл не существует или поврежден, возвращает настройки по умолчанию.
    """
    default_config = get_default_config()
    
    if not CONFIG_FILE.exists():
        return default_config
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        # Валидация и обновление конфига
        # Обновляем только валидные ключи из сохраненного конфига
        valid_keys = set(default_config.keys())
        filtered_config = {k: v for k, v in saved_config.items() if k in valid_keys}
        
        # Обновляем дефолтные значения сохраненными
        default_config.update(filtered_config)
        
        return default_config
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Ошибка парсинга конфига (неверный JSON): {e}")
        print(f"   Используем настройки по умолчанию")
        return default_config
    except IOError as e:
        print(f"⚠️ Ошибка чтения конфига: {e}")
        print(f"   Используем настройки по умолчанию")
        return default_config
    except Exception as e:
        print(f"⚠️ Неожиданная ошибка при загрузке конфига: {e}")
        print(f"   Используем настройки по умолчанию")
        return default_config


def save_config(config: Dict[str, Any]) -> None:
    """
    Сохраняет настройки в конфигурационный файл.
    
    Args:
        config: Словарь с настройками для сохранения
        
    Raises:
        IOError: Если не удалось записать файл
    """
    # Валидация и нормализация конфига
    validated_config = validate_config(config)
    
    # Сохраняем в файл
    try:
        # Проверяем, что CONFIG_FILE не является директорией (может случиться при неправильном монтировании в Docker)
        if CONFIG_FILE.exists() and CONFIG_FILE.is_dir():
            raise IOError(
                f"Ошибка: {CONFIG_FILE.absolute()} является директорией, а не файлом. "
                "Возможно, файл config.json не существует на хосте перед монтированием в Docker."
            )
        
        # Убеждаемся, что директория существует
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Используем системную временную директорию для создания временного файла
        # Это важно при работе с Docker volumes, где создание файла в той же директории может не работать
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            suffix='.json',
            delete=False,
            dir=tempfile.gettempdir()
        ) as temp_file:
            json.dump(validated_config, temp_file, indent=2, ensure_ascii=False)
            temp_path = temp_file.name
        
        # Копируем временный файл в целевое местоположение
        # Используем shutil.copy2 для сохранения метаданных
        try:
            shutil.copy2(temp_path, CONFIG_FILE)
        except (IOError, PermissionError, OSError) as e:
            # Если копирование не удалось, пытаемся записать напрямую
            try:
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(validated_config, f, indent=2, ensure_ascii=False)
            except (IOError, PermissionError, OSError) as direct_error:
                raise IOError(
                    f"Ошибка сохранения конфига в {CONFIG_FILE.absolute()}: "
                    f"копирование ({e}), прямая запись ({direct_error})"
                )
        finally:
            # Удаляем временный файл
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass  # Игнорируем ошибки при удалении временного файла
        
        print(f"✅ Конфиг сохранен в {CONFIG_FILE.absolute()}")
    except IOError:
        raise
    except Exception as e:
        raise IOError(f"Неожиданная ошибка сохранения конфига в {CONFIG_FILE.absolute()}: {e}")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Валидирует и нормализует конфиг, возвращая только валидные значения.
    Обрабатывает случаи, когда значения приходят как строки из форм.
    """
    default_config = get_default_config()
    validated = {}
    
    for key, default_value in default_config.items():
        if key in config:
            value = config[key]
            
            # Приводим к правильному типу
            try:
                if isinstance(default_value, bool):
                    # Обрабатываем строки "true"/"false" и другие варианты
                    if isinstance(value, str):
                        validated[key] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        validated[key] = bool(value)
                elif isinstance(default_value, int):
                    validated[key] = int(float(value))  # Сначала float, потом int (для "4.0" -> 4)
                elif isinstance(default_value, float):
                    validated[key] = float(value)
                elif isinstance(default_value, str):
                    validated[key] = str(value)
                else:
                    validated[key] = default_value
            except (ValueError, TypeError) as e:
                print(f"⚠️ Неверное значение для {key}: {value} ({type(value)}), используем значение по умолчанию: {default_value}")
                validated[key] = default_value
        else:
            validated[key] = default_value
    
    return validated
