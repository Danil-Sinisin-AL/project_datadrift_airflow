import json
import random
import pickle
import numpy as np
import csv
import os
from datetime import datetime
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Конфигурация MLflow
MLFLOW_TRACKING_URI = "file:/opt/airflow/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(MLFLOW_TRACKING_URI)

# Глобальные переменные для моделей
production_model = None
staging_model = None
traffic_split = 0.7  # 70% трафика на Production, 30% на Staging


class ModelLogger:
    """Класс для логирования запросов и предсказаний в CSV и JSON форматах"""

    def __init__(self, log_file='/opt/airflow/logs/model_predictions.log', 
                 csv_file='/opt/airflow/logs/model_predictions.csv'):
        self.log_file = log_file
        self.csv_file = csv_file
        # Убеждаемся, что директория существует
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Инициализируем CSV файл с заголовками, если файл не существует
        if not os.path.exists(self.csv_file):
            self._init_csv_file()

    def _init_csv_file(self):
        """Инициализация CSV файла с заголовками"""
        try:
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Заголовки CSV
                headers = [
                    'timestamp', 'user_id', 'model_version', 'model_stage', 
                    'prediction', 'probability', 'routing_method',
                    'request_ip', 'request_method', 'endpoint', 'features_json'
                ]
                writer.writerow(headers)
        except Exception as e:
            logger.warning(f"Не удалось инициализировать CSV файл: {e}")

    def log_prediction(self, user_id, model_version, model_stage,
                       features, prediction, probability, routing_method=None,
                       request_ip=None, request_method=None, endpoint=None):
        """Логирование предсказания в JSON и CSV форматах"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'user_id': user_id,
            'model_version': model_version,
            'model_stage': model_stage,
            'features': features,
            'prediction': int(prediction),
            'probability': float(probability),
            'routing_method': routing_method,
            'request_ip': request_ip,
            'request_method': request_method,
            'endpoint': endpoint
        }

        # Логируем в JSON файл
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Не удалось записать в JSON файл: {e}")

        # Логируем в CSV файл
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Основные поля
                row = [
                    timestamp,
                    user_id,
                    model_version,
                    model_stage,
                    int(prediction),
                    float(probability),
                    routing_method or '',
                    request_ip or '',
                    request_method or '',
                    endpoint or '',
                    json.dumps(features) if features else ''  # Фичи как JSON строка
                ]
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"Не удалось записать в CSV файл: {e}")

        # Логируем в MLflow для отслеживания (опционально, можно отключить для производительности)
        try:
            # Логируем только каждое N-е предсказание для снижения нагрузки
            if random.random() < 0.01:  # Логируем 1% предсказаний
                mlflow.set_experiment("ab_test_predictions")
                with mlflow.start_run(run_name=f"prediction_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_param("user_id", user_id)
                    mlflow.log_param("model_version", model_version)
                    mlflow.log_param("model_stage", model_stage)
                    mlflow.log_param("routing_method", routing_method)
                    mlflow.log_metric("prediction", int(prediction))
                    mlflow.log_metric("probability", float(probability))
        except Exception as e:
            logger.warning(f"Не удалось логировать в MLflow: {e}")

        logger.info(f"Logged prediction: model_version={model_version}, model_stage={model_stage}, "
                   f"prediction={prediction}, probability={probability:.4f}, routing={routing_method}")

    def log_request(self, endpoint, method, status_code, request_data=None, response_data=None, 
                   request_ip=None, user_id=None):
        """Логирование всех запросов к API"""
        timestamp = datetime.now().isoformat()
        request_file = '/opt/airflow/logs/api_requests.csv'
        
        # Инициализируем файл запросов, если не существует
        if not os.path.exists(request_file):
            try:
                with open(request_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'endpoint', 'method', 'status_code', 
                        'request_ip', 'user_id', 'request_data', 'response_data'
                    ])
            except Exception as e:
                logger.warning(f"Не удалось инициализировать файл запросов: {e}")
                return
        
        # Логируем запрос
        try:
            with open(request_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    endpoint,
                    method,
                    status_code,
                    request_ip or '',
                    user_id or '',
                    json.dumps(request_data) if request_data else '',
                    json.dumps(response_data) if response_data else ''
                ])
        except Exception as e:
            logger.warning(f"Не удалось записать запрос в CSV: {e}")


model_logger = ModelLogger()


# Flask hooks для логирования всех запросов
@app.before_request
def log_request_info():
    """Логирование информации о входящем запросе"""
    if request.path != '/health':  # Пропускаем health checks
        logger.debug(f"Request: {request.method} {request.path} from {request.remote_addr}")


@app.after_request
def log_response_info(response):
    """Логирование ответа после обработки запроса"""
    if request.path != '/health':  # Пропускаем health checks
        try:
            # Получаем данные запроса и ответа
            request_data = None
            response_data = None
            
            if request.is_json:
                try:
                    request_data = request.get_json()
                except:
                    pass
            
            if response.is_json:
                try:
                    response_data = response.get_json()
                except:
                    response_data = response.get_data(as_text=True)
            
            # Логируем запрос
            model_logger.log_request(
                endpoint=request.path,
                method=request.method,
                status_code=response.status_code,
                request_data=request_data,
                response_data=response_data,
                request_ip=request.remote_addr,
                user_id=request_data.get('user_id') if request_data and isinstance(request_data, dict) else None
            )
        except Exception as e:
            logger.warning(f"Не удалось залогировать запрос: {e}")
    
    return response


def load_models():
    """Загрузка Production и Staging моделей из MLflow Registry"""
    global production_model, staging_model
    model_name = "CreditRiskModel"

    try:
        # Загружаем Production модель
        try:
            prod_versions = client.search_model_versions(f"name='{model_name}' AND stage='Production'")
            if prod_versions:
                production_version = prod_versions[0]
                production_model = mlflow.pyfunc.load_model(
                    model_uri=f"models:/{model_name}/Production"
                )
                logger.info(f"Production модель загружена: версия {production_version.version}")
            else:
                logger.warning("Production модель не найдена")
                production_model = None
        except Exception as e:
            logger.warning(f"Ошибка загрузки Production модели: {e}")
            production_model = None

        # Загружаем Staging модель (если есть)
        try:
            staging_versions = client.search_model_versions(f"name='{model_name}' AND stage='Staging'")
            if staging_versions:
                staging_version = staging_versions[0]
                staging_model = mlflow.pyfunc.load_model(
                    model_uri=f"models:/{model_name}/Staging"
                )
                logger.info(f"Staging модель загружена: версия {staging_version.version}")
            else:
                logger.warning("Staging модель не найдена")
                staging_model = None
        except Exception as e:
            logger.warning(f"Ошибка загрузки Staging модели: {e}")
            staging_model = None

    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности API"""
    models_loaded = {
        'production': production_model is not None,
        'staging': staging_model is not None
    }
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'traffic_split': traffic_split,
        'production_percent': traffic_split * 100,
        'staging_percent': (1 - traffic_split) * 100
    }), 200


@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Перезагрузка моделей из MLflow Registry"""
    try:
        load_models()
        return jsonify({
            'status': 'success',
            'message': 'Models reloaded successfully',
            'models_loaded': {
                'production': production_model is not None,
                'staging': staging_model is not None
            }
        }), 200
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/update_split', methods=['POST'])
def update_traffic_split():
    """Динамическое обновление распределения трафика между моделями (например, 70/30, 50/50)"""
    global traffic_split

    try:
        data = request.get_json()
        new_split = float(data.get('production_split', 0.7))

        if 0 <= new_split <= 1:
            old_split = traffic_split
            traffic_split = new_split
            logger.info(
                f"Traffic split updated: {old_split * 100:.0f}% -> {traffic_split * 100:.0f}% production, "
                f"{(1 - traffic_split) * 100:.0f}% staging")

            return jsonify({
                'status': 'success',
                'old_split': old_split,
                'new_split': traffic_split,
                'production_percent': traffic_split * 100,
                'staging_percent': (1 - traffic_split) * 100,
                'message': f"Split updated to {traffic_split * 100:.0f}%/{100 - traffic_split * 100:.0f}%"
            }), 200
        else:
            return jsonify({'error': 'Split must be between 0 and 1'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_split', methods=['GET'])
def get_traffic_split():
    """Получить текущее распределение трафика"""
    return jsonify({
        'production_split': traffic_split,
        'production_percent': traffic_split * 100,
        'staging_percent': (1 - traffic_split) * 100,
        'split_ratio': f"{traffic_split * 100:.0f}/{100 - traffic_split * 100:.0f}"
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Основной endpoint для предсказаний с A/B тестированием"""
    try:
        # Получаем данные запроса
        data = request.get_json()

        # Извлекаем user_id (если есть) или генерируем случайный
        user_id = data.get('user_id', random.randint(1, 1000000))

        # Извлекаем фичи для предсказания
        features = data.get('features', {})

        # Проверяем, что есть хотя бы Production модель
        if production_model is None:
            return jsonify({'error': 'Production model not loaded'}), 500

        # Определяем, какую модель использовать
        # Метод 1: по user_id (user_id % 2 == 0 -> Production, иначе -> Staging)
        # Метод 2: случайное распределение с учетом traffic_split
        use_production = True
        
        if 'user_id' in data:
            # Детерминированное распределение по user_id
            use_production = (user_id % 2 == 0)
            routing_method = "user_id_based"
        else:
            # Случайное распределение с учетом traffic_split
            use_production = random.random() < traffic_split
            routing_method = "random_split"

        # Выбираем модель
        if use_production and production_model:
            model = production_model
            model_stage = "Production"

            # Получаем информацию о версии Production модели
            prod_version = client.get_latest_versions(
                "CreditRiskModel",
                stages=["Production"]
            )[0]
            model_version = prod_version.version

        elif staging_model:
            model = staging_model
            model_stage = "Staging"

            # Получаем информацию о версии Staging модели
            staging_version = client.get_latest_versions(
                "CreditRiskModel",
                stages=["Staging"]
            )[0]
            model_version = staging_version.version
        else:
            # Если нет Staging, используем Production
            model = production_model
            model_stage = "Production"
            prod_version = client.get_latest_versions(
                "CreditRiskModel",
                stages=["Production"]
            )[0]
            model_version = prod_version.version

        # Подготавливаем данные для предсказания
        features_array = np.array([list(features.values())])

        # Делаем предсказание
        prediction = model.predict(features_array)
        probabilities = model.predict_proba(features_array)

        # Логируем предсказание (версия модели, input-фичи, результат предсказания)
        model_logger.log_prediction(
            user_id=user_id,
            model_version=model_version,
            model_stage=model_stage,
            features=features,
            prediction=int(prediction[0]),
            probability=float(probabilities[0][1]),
            routing_method=routing_method,
            request_ip=request.remote_addr,
            request_method=request.method,
            endpoint=request.path
        )
        
        # Запрос уже логируется через @app.after_request, но можно добавить дополнительное логирование

        # Формируем ответ
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probabilities[0][1]),
            'model_version': model_version,
            'model_stage': model_stage,
            'used_production': use_production,
            'user_id': user_id,
            'routing_method': routing_method,
            'traffic_split': traffic_split if routing_method == "random_split" else None
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/promote_to_production', methods=['POST'])
def promote_to_production():
    """Перевод Staging модели в Production"""
    try:
        data = request.get_json()

        # Проверяем A/B тест метрики (здесь можно добавить реальную логику проверки)
        a_b_test_passed = data.get('ab_test_passed', False)

        if not a_b_test_passed:
            return jsonify({
                'error': 'A/B test not passed',
                'message': 'Cannot promote model without successful A/B test'
            }), 400

        # Получаем Staging модель
        staging_version = client.get_latest_versions(
            "CreditRiskModel",
            stages=["Staging"]
        )[0]

        # Архивируем текущую Production модель
        try:
            prod_version = client.get_latest_versions(
                "CreditRiskModel",
                stages=["Production"]
            )[0]

            client.transition_model_version_stage(
                name="CreditRiskModel",
                version=prod_version.version,
                stage="Archived"
            )
            logger.info(f"Production model v{prod_version.version} archived")
        except IndexError:
            logger.info("No existing Production model found")

        # Переводим Staging в Production
        client.transition_model_version_stage(
            name="CreditRiskModel",
            version=staging_version.version,
            stage="Production"
        )

        # Обновляем загруженные модели
        load_models()

        return jsonify({
            'status': 'success',
            'message': f'Model v{staging_version.version} promoted to Production',
            'new_production_version': staging_version.version
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Загружаем модели при старте
    load_models()

    # Запускаем Flask приложение
    app.run(host='0.0.0.0', port=5000, debug=True)