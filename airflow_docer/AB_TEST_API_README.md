# Flask A/B Test API

Минимальный Flask API-роутер для A/B тестирования моделей Production (A) и Staging (B).

## Endpoints

### 1. `GET /health`
Проверка работоспособности API и статуса загруженных моделей.

**Ответ:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "production": true,
    "staging": true
  },
  "traffic_split": 0.7,
  "production_percent": 70.0,
  "staging_percent": 30.0
}
```

### 2. `POST /predict`
Основной endpoint для предсказаний с A/B тестированием.

**Запрос:**
```json
{
  "user_id": 12345,  // Опционально: для детерминированного распределения
  "features": {
    "feature1": 0.5,
    "feature2": 0.3,
    ...
  }
}
```

**Ответ:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "model_version": "2",
  "model_stage": "Production",
  "used_production": true,
  "user_id": 12345,
  "routing_method": "user_id_based",
  "traffic_split": null
}
```

**Логика маршрутизации:**
- Если передан `user_id`: используется детерминированное распределение `user_id % 2 == 0` → Production, иначе → Staging
- Если `user_id` не передан: используется случайное распределение с учетом `traffic_split` (по умолчанию 70/30)

### 3. `POST /update_split`
Динамическое обновление распределения трафика между моделями.

**Запрос:**
```json
{
  "production_split": 0.5  // 50% на Production, 50% на Staging
}
```

**Ответ:**
```json
{
  "status": "success",
  "old_split": 0.7,
  "new_split": 0.5,
  "production_percent": 50.0,
  "staging_percent": 50.0,
  "message": "Split updated to 50.0%/50.0%"
}
```

### 4. `GET /get_split`
Получить текущее распределение трафика.

**Ответ:**
```json
{
  "production_split": 0.7,
  "production_percent": 70.0,
  "staging_percent": 30.0,
  "split_ratio": "70/30"
}
```

### 5. `POST /reload_models`
Перезагрузка моделей из MLflow Registry (полезно после обновления моделей).

**Ответ:**
```json
{
  "status": "success",
  "message": "Models reloaded successfully",
  "models_loaded": {
    "production": true,
    "staging": true
  }
}
```

## Логирование

API логирует следующие данные для каждого предсказания:
- **Версия модели** (`model_version`)
- **Стадия модели** (`model_stage`: Production/Staging)
- **Input-фичи** (`features`)
- **Результат предсказания** (`prediction`, `probability`)
- **Метод маршрутизации** (`routing_method`: user_id_based/random_split)
- **User ID** (`user_id`)
- **Timestamp** (`timestamp`)

Логи сохраняются в:
- Файл: `/opt/airflow/logs/model_predictions.log`
- MLflow: эксперимент `ab_test_predictions` (1% предсказаний для снижения нагрузки)

## Примеры использования

### Пример 1: Предсказание с user_id (детерминированное распределение)
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "features": {
      "feature1": 0.5,
      "feature2": 0.3
    }
  }'
```

### Пример 2: Предсказание без user_id (случайное распределение)
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature1": 0.5,
      "feature2": 0.3
    }
  }'
```

### Пример 3: Изменение распределения трафика на 50/50
```bash
curl -X POST http://localhost:5001/update_split \
  -H "Content-Type: application/json" \
  -d '{
    "production_split": 0.5
  }'
```

### Пример 4: Проверка статуса
```bash
curl http://localhost:5001/health
```

## Порты

- **Flask A/B Test API**: `http://localhost:5001`
- **MLflow UI**: `http://localhost:5000`
- **Airflow UI**: `http://localhost:8080`

## Интеграция с DAG

После переобучения модели в DAG (`retrain_task`), новая модель автоматически попадает в стадию Staging и становится доступной для A/B тестирования через Flask API.

После успешного A/B теста модель может быть автоматически переведена в Production через задачу `promote_to_production_task` в DAG.
