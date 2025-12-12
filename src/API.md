# Portrait Processing API - Спецификация

## Базовый URL

```
http://localhost:8000
```

## Значения по умолчанию

API использует следующие значения по умолчанию для всех параметров:

| Параметр | Значение по умолчанию |
|----------|----------------------|
| `face_enhance` | `true` (включено) |
| `face_upscale` | `4` |
| `face_strength` | `0.7` |
| `face_iterations` | `2` |
| `alpha_erode` | `10` |
| `alpha_dilate` | `0` |
| `alpha_feather` | `10` |
| `bg_model` | `u2net_human_seg` |
| `keep_largest` | `true` |
| `reference` | `src/ref.png` (если файл существует) |
| `background` | `src/bg.jpg` (если файл существует) |

**Важно:**
- Референс (`reference`) и фон (`background`) опциональны - если не указаны, используются файлы по умолчанию
- Итоговое изображение всегда имеет размер **720x1280 пикселей** с сохранением пропорций

## Endpoints

### 1. GET `/api/health`

Проверка работоспособности API.

**Запрос:**
```http
GET /api/health HTTP/1.1
Host: localhost:8000
```

**Ответ:**
```json
{
  "status": "ok",
  "message": "API is running"
}
```

**Статус коды:**
- `200 OK` - API работает

---

### 2. POST `/api/process`

Обработка портрета: удаление фона, перенос цвета, улучшение лица, наложение фона.

**Запрос:**
```http
POST /api/process HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="portrait"; filename="portrait.jpg"
Content-Type: image/jpeg

[бинарные данные изображения]
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="reference"; filename="reference.png"
Content-Type: image/png

[бинарные данные изображения]
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="background"; filename="background.jpg"
Content-Type: image/jpeg

[бинарные данные изображения]
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="face_enhance"

true
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="face_upscale"

3
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="alpha_erode"

2
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="alpha_dilate"

2
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="alpha_feather"

5
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

**Параметры (multipart/form-data):**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `portrait` | File | ✅ Да | Исходное изображение портрета (JPG, PNG, WEBP) |
| `reference` | File | ❌ Нет | Референсное изображение для переноса цвета. Если не указано, используется `src/ref.png` по умолчанию |
| `background` | File | ❌ Нет | Фоновое изображение (будет подложено под результат). Если не указано, используется `src/bg.jpg` по умолчанию (если файл существует) |
| `face_enhance` | Boolean | ❌ Нет | Включить улучшение лица через GFPGAN (по умолчанию: `true`) |
| `face_upscale` | Integer | ❌ Нет | Масштаб улучшения GFPGAN: 1-4 (по умолчанию: `4`, рекомендуется 2-4) |
| `face_strength` | Float | ❌ Нет | Интенсивность улучшения: 0.0-1.0 (по умолчанию: `0.7`, где 1.0 = полный эффект) |
| `face_iterations` | Integer | ❌ Нет | Количество итераций улучшения: 1-3 (по умолчанию: `2`, больше = сильнее эффект) |
| `alpha_erode` | Integer | ❌ Нет | Эрозия маски в пикселях: 0-10 (по умолчанию: `10`) |
| `alpha_dilate` | Integer | ❌ Нет | Дилятация маски в пикселях: 0-10 (по умолчанию: `0`) |
| `alpha_feather` | Integer | ❌ Нет | Размытие края маски в пикселях: 0-10 (по умолчанию: `10`) |
| `keep_largest` | Boolean | ❌ Нет | Оставить только главного человека (убрать людей на фоне) (по умолчанию: `true`) |
| `bg_model` | String | ❌ Нет | Модель для удаления фона (по умолчанию: `u2net_human_seg`) |

**Ответ:**

**Успешный ответ (200 OK):**
- **Content-Type:** `image/png`
- **Content-Disposition:** `attachment; filename=result.png`
- **X-Processing-Time:** Время обработки в секундах (например: `12.34`)
- **Тело:** Бинарные данные PNG изображения

**Важно:**
- Итоговое изображение всегда имеет размер **720x1280 пикселей**
- Изображение вписывается в этот размер с сохранением пропорций (без растягивания)
- Если исходное изображение меньше целевого размера, добавляется прозрачный фон
- Изображение центрируется в итоговом размере

**Ошибки:**

**400 Bad Request:**
```json
{
  "detail": "Портрет не может быть пустым"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Описание ошибки обработки"
}
```

---

## Примеры использования

### cURL

#### Базовый запрос (используются файлы по умолчанию)
```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "portrait=@portrait.jpg" \
  -o result.png
```
*Примечание: Если не указаны `reference` и `background`, используются `src/ref.png` и `src/bg.jpg` по умолчанию*

#### С кастомным референсом
```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "portrait=@portrait.jpg" \
  -F "reference=@reference.png" \
  -o result.png
```

#### С настройками по умолчанию (явно указаны)
```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "portrait=@portrait.jpg" \
  -F "face_enhance=true" \
  -F "face_upscale=4" \
  -F "face_strength=0.7" \
  -F "face_iterations=2" \
  -F "alpha_erode=10" \
  -F "alpha_feather=10" \
  -F "bg_model=u2net_human_seg" \
  -o result.png
```

#### Полная настройка
```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "portrait=@portrait.jpg" \
  -F "reference=@reference.png" \
  -F "background=@background.jpg" \
  -F "face_enhance=true" \
  -F "face_upscale=4" \
  -F "face_strength=0.7" \
  -F "face_iterations=2" \
  -F "alpha_erode=10" \
  -F "alpha_dilate=0" \
  -F "alpha_feather=10" \
  -F "bg_model=u2net_human_seg" \
  -F "keep_largest=true" \
  -o result.png
```

### Python (requests)

```python
import requests

url = "http://localhost:8000/api/process"

files = {
    'portrait': ('portrait.jpg', open('portrait.jpg', 'rb'), 'image/jpeg'),
    'reference': ('reference.png', open('reference.png', 'rb'), 'image/png'),
    'background': ('background.jpg', open('background.jpg', 'rb'), 'image/jpeg'),
}

data = {
    'face_enhance': True,
    'face_upscale': 4,
    'face_strength': 0.7,
    'face_iterations': 2,
    'alpha_erode': 10,
    'alpha_dilate': 0,
    'alpha_feather': 10,
    'bg_model': 'u2net_human_seg',
    'keep_largest': True,
}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open('result.png', 'wb') as f:
        f.write(response.content)
    print("Результат сохранен в result.png")
else:
    print(f"Ошибка: {response.status_code}")
    print(response.json())
```

### Python (httpx - async)

```python
import httpx
import asyncio

async def process_image():
    url = "http://localhost:8000/api/process"
    
    files = {
        'portrait': ('portrait.jpg', open('portrait.jpg', 'rb'), 'image/jpeg'),
        'reference': ('reference.png', open('reference.png', 'rb'), 'image/png'),
    }
    
    data = {
        'face_enhance': True,
        'face_upscale': 4,
        'face_strength': 0.7,
        'face_iterations': 2,
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, files=files, data=data)
        
        if response.status_code == 200:
            with open('result.png', 'wb') as f:
                f.write(response.content)
            print("Результат сохранен")
        else:
            print(f"Ошибка: {response.status_code}")

asyncio.run(process_image())
```

### JavaScript (fetch)

```javascript
async function processImage() {
    const formData = new FormData();
    
    // Добавляем файлы
    const portraitFile = document.getElementById('portraitInput').files[0];
    const referenceFile = document.getElementById('referenceInput').files[0];
    const backgroundFile = document.getElementById('backgroundInput').files[0];
    
    formData.append('portrait', portraitFile);
    formData.append('reference', referenceFile);
    if (backgroundFile) {
        formData.append('background', backgroundFile);
    }
    
    // Добавляем параметры (значения по умолчанию)
    formData.append('face_enhance', true);
    formData.append('face_upscale', 4);
    formData.append('face_strength', 0.7);
    formData.append('face_iterations', 2);
    formData.append('alpha_erode', 10);
    formData.append('alpha_dilate', 0);
    formData.append('alpha_feather', 10);
    formData.append('bg_model', 'u2net_human_seg');
    formData.append('keep_largest', true);
    
    try {
        const response = await fetch('http://localhost:8000/api/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка обработки');
        }
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        // Скачиваем файл
        const a = document.createElement('a');
        a.href = url;
        a.download = 'result.png';
        a.click();
        
        console.log('Результат скачан');
    } catch (error) {
        console.error('Ошибка:', error);
    }
}
```

### Node.js (axios)

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function processImage() {
    const form = new FormData();
    
    form.append('portrait', fs.createReadStream('portrait.jpg'));
    form.append('reference', fs.createReadStream('reference.png'));
    form.append('background', fs.createReadStream('background.jpg'));
    form.append('face_enhance', 'true');
    form.append('face_upscale', '4');
    form.append('face_strength', '0.7');
    form.append('face_iterations', '2');
    form.append('alpha_erode', '10');
    form.append('alpha_dilate', '0');
    form.append('alpha_feather', '10');
    form.append('bg_model', 'u2net_human_seg');
    
    try {
        const response = await axios.post(
            'http://localhost:8000/api/process',
            form,
            {
                headers: form.getHeaders(),
                responseType: 'arraybuffer',
                timeout: 300000 // 5 минут
            }
        );
        
        fs.writeFileSync('result.png', response.data);
        console.log('Результат сохранен в result.png');
    } catch (error) {
        console.error('Ошибка:', error.response?.data || error.message);
    }
}

processImage();
```

### PHP

```php
<?php
$url = 'http://localhost:8000/api/process';

$data = [
    'portrait' => new CURLFile('portrait.jpg', 'image/jpeg', 'portrait.jpg'),
    'reference' => new CURLFile('reference.png', 'image/png', 'reference.png'),
    'background' => new CURLFile('background.jpg', 'image/jpeg', 'background.jpg'),
    'face_enhance' => 'true',
    'face_upscale' => '4',
    'face_strength' => '0.7',
    'face_iterations' => '2',
    'alpha_erode' => '10',
    'alpha_dilate' => '0',
    'alpha_feather' => '10',
    'bg_model' => 'u2net_human_seg',
];

$ch = curl_init($url);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_TIMEOUT, 300);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($httpCode === 200) {
    file_put_contents('result.png', $response);
    echo "Результат сохранен в result.png\n";
} else {
    echo "Ошибка: HTTP $httpCode\n";
    echo $response;
}
?>
```

### Go

```go
package main

import (
    "bytes"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
)

func processImage() error {
    url := "http://localhost:8000/api/process"
    
    var b bytes.Buffer
    w := multipart.NewWriter(&b)
    
    // Добавляем файлы
    portraitFile, _ := os.Open("portrait.jpg")
    defer portraitFile.Close()
    portraitWriter, _ := w.CreateFormFile("portrait", "portrait.jpg")
    io.Copy(portraitWriter, portraitFile)
    
    referenceFile, _ := os.Open("reference.png")
    defer referenceFile.Close()
    referenceWriter, _ := w.CreateFormFile("reference", "reference.png")
    io.Copy(referenceWriter, referenceFile)
    
    // Добавляем параметры
    w.WriteField("face_enhance", "true")
    w.WriteField("face_upscale", "4")
    w.WriteField("face_strength", "0.7")
    w.WriteField("face_iterations", "2")
    w.WriteField("alpha_erode", "10")
    w.WriteField("alpha_dilate", "0")
    w.WriteField("alpha_feather", "10")
    w.WriteField("bg_model", "u2net_human_seg")
    
    w.Close()
    
    req, _ := http.NewRequest("POST", url, &b)
    req.Header.Set("Content-Type", w.FormDataContentType())
    
    client := &http.Client{Timeout: 300 * time.Second}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != 200 {
        body, _ := io.ReadAll(resp.Body)
        return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
    }
    
    out, _ := os.Create("result.png")
    defer out.Close()
    io.Copy(out, resp.Body)
    
    fmt.Println("Результат сохранен в result.png")
    return nil
}

func main() {
    if err := processImage(); err != nil {
        fmt.Fprintf(os.Stderr, "Ошибка: %v\n", err)
    }
}
```

---

## Ограничения и рекомендации

### Размер файлов
- Рекомендуемый максимальный размер изображения: **10 МБ**
- Максимальный размер запроса: зависит от настроек сервера (по умолчанию ~100 МБ)

### Таймауты
- Рекомендуемый таймаут клиента: **300 секунд (5 минут)**
- Обработка с GFPGAN может занимать 30-120 секунд в зависимости от размера изображения

### Поддерживаемые форматы
- **Входные изображения:** JPG, JPEG, PNG, WEBP
- **Выходное изображение:** PNG (с прозрачностью)

### Параметры обработки

**bg_model (модель удаления фона):**
- `u2net_human_seg` - **U2Net для людей** (по умолчанию, хорошо для портретов, быстрее ISNet)
- `isnet-general-use` - ISNet (лучшая точность, рекомендуется для портретов) ⭐
- `u2net` - U2Net базовая (универсальная, быстрая)
- `silueta` - Silueta (хорошая для общих случаев)
- `u2netp` - U2Net Lite (самая быстрая, но менее точная)

**face_upscale:**
- `1` - минимальная обработка (быстро)
- `2-3` - оптимальный баланс качества и скорости
- `4` - максимальное качество (медленно, по умолчанию)

**face_strength (интенсивность улучшения):**
- `0.0` - без улучшения (исходное изображение)
- `0.3-0.7` - легкое улучшение (естественный вид, по умолчанию: `0.7`)
- `1.0` - полное улучшение (максимальный эффект)

**face_iterations (количество итераций):**
- `1` - одно применение улучшения (быстро)
- `2` - двойное применение (более сильный эффект, по умолчанию)
- `3` - тройное применение (максимальный эффект, медленно)

**alpha_erode / alpha_dilate / alpha_feather:**
- `0` - без обработки краев
- `1-3` - легкая обработка
- `4-7` - средняя обработка
- `8-10` - сильная обработка (может обрезать детали)
- По умолчанию: `alpha_erode=10`, `alpha_dilate=0`, `alpha_feather=10`

---

## OpenAPI / Swagger документация

FastAPI автоматически генерирует OpenAPI спецификацию:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## Коды ошибок

| Код | Описание |
|-----|----------|
| `200` | Успешная обработка |
| `400` | Неверные параметры запроса (отсутствуют обязательные файлы) |
| `422` | Ошибка валидации данных |
| `500` | Внутренняя ошибка сервера (ошибка обработки изображения) |

---

## Примечания

1. Все изображения обрабатываются в памяти, временные файлы не сохраняются на диск
2. Результат всегда возвращается в формате PNG с альфа-каналом
3. **Итоговое изображение всегда имеет размер 720x1280 пикселей** с сохранением пропорций исходного изображения
4. Если указан `background`, он автоматически масштабируется под размер портрета
5. Если `reference` не указан, используется `src/ref.png` по умолчанию (если файл существует)
6. Если `background` не указан, используется `src/bg.jpg` по умолчанию (если файл существует)
7. GFPGAN требует значительных ресурсов - рекомендуется использовать на серверах с GPU для ускорения
8. Время обработки возвращается в заголовке ответа `X-Processing-Time`

