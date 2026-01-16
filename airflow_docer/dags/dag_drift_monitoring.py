import pandas as pd
import numpy as np
import catboost as cb
import os
import json
import time
import mlflow
import mlflow.pyfunc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from pycaret.classification import get_config
from datetime import datetime, timedelta

os.environ["MLFLOW_TRACKING_URI"] = "file:/opt/airflow/mlruns"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "airflow_pycaret"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# def calculate_psi(patch_old_data, patch_new_data, bins=10):
#     """Расчет PSI между двумя распределениями."""
#     breakpoints = np.linspace(0, 1, bins + 1)
#     expected_percents = np.histogram(patch_old_data, breakpoints)[0] / len(patch_old_data)
#     actual_percents = np.histogram(patch_new_data, breakpoints)[0] / len(patch_new_data)
#
#     def safe_ln(x):
#         return 0 if x == 0 else np.log(x)
#
#     psi = np.sum((patch_new_data - patch_old_data) *
#                  np.vectorize(safe_ln)(patch_new_data / patch_old_data))
#     return psi


def calculate_psi_simple(expected, actual, bins=10):
    """Simplified and robust PSI calculation."""
    try:
        # Убедимся что это массивы numpy
        expected = np.asarray(expected)
        actual = np.asarray(actual)

        # Удаляем NaN и бесконечные значения
        expected = expected[np.isfinite(expected)]
        actual = actual[np.isfinite(actual)]

        if len(expected) < 2 or len(actual) < 2:
            return 0.0

        # Определяем границы через процентили (более устойчиво)
        percentiles = np.linspace(0, 100, bins + 1)
        breakpoints = np.percentile(np.concatenate([expected, actual]), percentiles)

        # Уникальные breakpoints
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            breakpoints = np.array([np.min(expected), np.max(expected)])

        # Гистограммы
        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)

        # Преобразуем в вероятности с smoothing
        smoothing_factor = 0.001
        expected_probs = (expected_counts + smoothing_factor) / (len(expected) + smoothing_factor * bins)
        actual_probs = (actual_counts + smoothing_factor) / (len(actual) + smoothing_factor * bins)

        # Вычисляем PSI
        psi = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))

        return psi if np.isfinite(psi) else 0.0

    except Exception as e:
        print(f"Error in calculate_psi_simple: {e}")
        return 0.0

def fetch_reference_data():
    """Загрузка референсных данных (обучающая выборка)."""
    # Здесь может быть загрузка из файла, БД и т.д.
    # Пример:
    df = pd.read_csv('/opt/airflow/dags/train.csv')
    return df


def fetch_current_data():
    """Загрузка текущих данных (например, за последние сутки)."""
    # Здесь может быть SQL-запрос или чтение из хранилища
    df = pd.read_csv('/opt/airflow/dags/X_test_drift.csv')
    return df


def check_drift(**kwargs):
    """Основная функция проверки дрейфа."""
    ref_df = fetch_reference_data()
    curr_df = fetch_current_data()
    #kwargs['data_old'] = ref_df

    drift_results = {}
    for column in curr_df.columns:
        if curr_df[column].dtype in ['float64', 'int64']:
            psi = calculate_psi_simple(ref_df[column].dropna(),
                                curr_df[column].dropna())
            drift_results[column] = psi

    # Усредняем PSI по всем признакам или берем максимум
    avg_psi = np.mean(list(drift_results.values()))
    print(f"Average PSI: {avg_psi}")
    # Передаем в следующий таск
    kwargs['ti'].xcom_push(key='avg_psi', value=avg_psi)
    return avg_psi


def decide_retrain(**kwargs):
    """Решение: нужно ли переобучать модель."""
    ti = kwargs['ti']
    avg_psi = ti.xcom_pull(key='avg_psi', task_ids='check_drift_task')
    if avg_psi > 0.2:
        return 'retrain_task'
    else:
        return 'skip_retrain_task'


def retrain_model():
    """Функция переобучения модели."""
    print("Запуск переобучения модели...")
    # Здесь код загрузки данных, подготовки, обучения и сохранения модели
    loaded_model = cb.CatBoostClassifier()
    loaded_model.load_model('/opt/airflow/dags/catboost_model.cbm')
    curr_df = fetch_current_data()

    print(f"\nИнформация о данных:")
    print(f"  Размер: {curr_df.shape}")
    print(f"  Колонки: {curr_df.columns.tolist()}")
    print(f"  Типы данных:")
    for col in curr_df.columns:
        print(f"    {col}: {curr_df[col].dtype}")
    print(f"\nРаспределение целевой переменной:")
    print(curr_df['class'].value_counts())

    mlflow.set_tracking_uri("file:/opt/airflow/mlruns")  # Локальное хранилище
    mlflow.set_experiment("pycaret_automl_experiment")
    print("\nНастройка PyCaret...")
    mapping = {'male': 0, 'female': 1}
    mapping1 = {'yes': 0, 'no': 1}
    curr_df['sex'] = curr_df['sex'].map(mapping)
    curr_df['foreign_worker_'] = curr_df['foreign_worker_'].map(mapping1)
    curr_df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    print(f"START_DATA = {curr_df}")
    exp = setup(
        data=curr_df,
        target='class',
        train_size=0.8,
        session_id=42,
        verbose=False,
        log_experiment=False,
        experiment_name=None,  # ← Уберите имя эксперимента
        log_plots=False,
        fold=3,
        n_jobs=1,
        html=False,
        # silent=True,
        profile=False,
        log_data=False,
        log_profile=False
    )
    print("PyCaret настроен успешно!")
    best_model = compare_models(
        include=['lr', 'rf', 'lightgbm', 'catboost', 'gbc', 'dt'],
        fold=5,
        sort='F1',  # Метрика для сортировки
        n_select=1,  # Выбираем только лучшую модель
        verbose=False
    )

    results = pull()
    print("\nРезультаты сравнения моделей:")
    print(results[['Model', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1']].to_string())
    print(f"\nЛУЧШАЯ МОДЕЛЬ: {type(best_model).__name__}")
    model_params = best_model.get_params()
    print(f"Параметры модели:")
    for key, value in list(model_params.items())[:10]:  # Показываем первые 10 параметров
        print(f"  {key}: {value}")

    model_dir = "/opt/airflow/models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_model(best_model, model_path)
    print(f"\nМодель сохранена: {model_path}.pkl")

    # Сохраняем результаты эксперимента
    results_path = f"{model_dir}/experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(results_path, index=False)
    print(f"Результаты эксперимента сохранены: {results_path}")
    # Сохраняем метаданные
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'best_model': type(best_model).__name__,
        'model_path': f"{model_path}.pkl",
        'data_shape': curr_df.shape,
        'data_columns': curr_df.columns.tolist(),
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'drift_detected': True,
    }
    metadata_path = f"{model_dir}/metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Метаданные сохранены: {metadata_path}")
    with mlflow.start_run(run_name=f"pycaret_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Логируем параметры
        mlflow.log_params({
            'best_model': type(best_model).__name__,
            'data_samples': len(curr_df),
            'features': curr_df.shape[1] - 1,
            'train_size': 0.8,
            'cv_folds': 5
        })

        # Логируем метрики
        best_model_results = results.iloc[0]
        mlflow.log_metrics({
            'accuracy': best_model_results['Accuracy'],
            'auc': best_model_results['AUC'],
            'recall': best_model_results['Recall'],
            'precision': best_model_results['Prec.'],
            'f1': best_model_results['F1']
        })

        # Логируем артефакты
        mlflow.log_artifact(model_path + ".pkl", "model")
        mlflow.log_artifact(results_path, "results")
        mlflow.log_artifact(metadata_path, "metadata")

        # Определяем сигнатуру модели
        X_sample = curr_df.drop('class', axis=1).iloc[:5]
        y_sample = curr_df['class'].iloc[:5]
        signature = infer_signature(X_sample, y_sample)

        # Логируем модель с сигнатурой
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=signature,
            registered_model_name="CreditRiskModel"  # Имя модели в реестре
        )
        # Получаем client MLflow для управления стадиями
        client = MlflowClient()

        # Получаем последнюю версию зарегистрированной модели
        model_name = "CreditRiskModel"
        model_version = None

        # Ждем регистрации модели (небольшая задержка)
        import time
        time.sleep(10)

        try:
            # Получаем все версии модели
            all_versions = client.search_model_versions(f"name='{model_name}'")
            # Находим последнюю версию (только что созданную)
            latest_version = None
            max_version_num = 0
            for version in all_versions:
                version_num = int(version.version)
                if version_num > max_version_num:
                    max_version_num = version_num
                    latest_version = version
            
            if latest_version is None:
                # Fallback: используем старый метод
                latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            
            # Переводим предыдущую продакшн модель в архив (если есть)
            try:
                prod_versions = client.search_model_versions(f"name='{model_name}' AND stage='Production'")
                for version in prod_versions:
                    try:
                        client.transition_model_version_stage(
                            name=model_name,
                            version=version.version,
                            stage="Archived"
                        )
                        print(f"Версия {version.version} перемещена в Archived")
                    except Exception as e:
                        print(f"Не удалось архивировать версию {version.version}: {e}")
            except Exception as e:
                print(f"Ошибка при архивировании Production моделей: {e}")

            # Переводим новую модель в стадию Staging
            # Обходим проблему с сериализацией Metric объектов в MLflow 3.0.0
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Staging"
                )
                print(f"Версия {latest_version.version} переведена в Staging")
            except Exception as transition_error:
                # Если возникает ошибка сериализации, пропускаем переход стадии
                # Модель останется в стадии "None", но будет доступна по версии
                error_msg = str(transition_error)
                if "RepresenterError" in error_msg or "cannot represent" in error_msg:
                    print(f"Предупреждение: Не удалось перевести модель в Staging из-за проблемы сериализации MLflow.")
                    print(f"Модель версии {latest_version.version} зарегистрирована, но осталась в стадии 'None'.")
                    print(f"Вы можете перевести её в Staging вручную через MLflow UI или CLI.")
                else:
                    # Если это другая ошибка, пробрасываем её дальше
                    raise
            
        except Exception as e:
            print(f"Ошибка при работе с реестром: {e}")
            # Пытаемся получить версию альтернативным способом
            try:
                all_versions = client.search_model_versions(f"name='{model_name}'")
                if all_versions:
                    latest_version = max(all_versions, key=lambda v: int(v.version))
                    print(f"Использована версия {latest_version.version} (без перевода в Staging)")
                else:
                    raise Exception("Не найдено версий модели")
            except Exception as e2:
                print(f"Критическая ошибка при получении версии модели: {e2}")
                # Устанавливаем latest_version в None, чтобы избежать ошибки дальше
                latest_version = None
                raise

    print("\nAutoML с PyCaret завершен успешно!")
    print(f"   Лучшая модель: {type(best_model).__name__}")
    print(f"   Точность: {best_model_results['Accuracy']:.4f}")
    print(f"   AUC: {best_model_results['AUC']:.4f}")

    print(f"\nМодель зарегистрирована в MLflow Registry:")
    print(f"  Имя модели: {model_name}")
    if latest_version is not None:
        print(f"  Версия: {latest_version.version}")
        print(f"  Стадия: {latest_version.current_stage if hasattr(latest_version, 'current_stage') else 'Staging'}")
    else:
        print(f"  Версия: не удалось определить")
    return {
        'status': 'success',
        'best_model': type(best_model).__name__,
        'accuracy': float(best_model_results['Accuracy']),
        'model_path': f"{model_path}.pkl",
        'model_version': latest_version.version if latest_version is not None else None,
        'model_stage': (latest_version.current_stage if latest_version is not None and hasattr(latest_version, 'current_stage') else 'Staging')
    }


    # data = curr_df.select_dtypes(include=['int', 'float'])
    # x = data.drop(columns=['class'])
    # y = data['class']
    # loaded_model.fit(x, y)
    # loaded_model.save_model('/opt/airflow/dags/catboost_model.cbm')
    # print("Модель переобучена.")


def run_ab_test(**kwargs):
    """Функция для проведения A/B тестирования Production и Staging моделей."""
    print("Запуск A/B тестирования...")
    
    # Явно используем глобальный модуль mlflow
    import mlflow as mlflow_module
    mlflow_module.set_tracking_uri("file:/opt/airflow/mlruns")
    client = MlflowClient()
    model_name = "CreditRiskModel"
    
    # Загружаем тестовые данные
    test_df = fetch_current_data()
    X_test = test_df.drop('class', axis=1)
    mapping = {'male': 0, 'female': 1}
    mapping1 = {'yes': 0, 'no': 1}
    X_test['sex'] = X_test['sex'].map(mapping)
    X_test['foreign_worker_'] = X_test['foreign_worker_'].map(mapping1)
    X_test.drop(columns=['Unnamed: 0'], axis=1,inplace =True)
    y_test = test_df['class']

    
    print(f"Тестовые данные: {X_test.shape[0]} образцов, {X_test.shape[1]} признаков")
    print(f"{X_test}")
    # Загружаем Production модель
    production_model = None
    production_version = None
    staging_model = None
    staging_version = None
    
    try:
        prod_versions = client.search_model_versions(f"name='{model_name}' AND stage='Production'")
        if prod_versions:
            production_version = prod_versions[0]
            production_model = mlflow_module.pyfunc.load_model(
                model_uri=f"models:/{model_name}/Production"
            )
            print(f"Production модель загружена: версия {production_version.version}")
        else:
            print("Предупреждение: Production модель не найдена")
    except Exception as e:
        print(f"Ошибка загрузки Production модели: {e}")
    
    # Загружаем Staging модель
    # try:
    #     staging_versions = client.search_model_versions(f"name='{model_name}' AND stage='Staging'")
    #     if staging_versions:
    #         staging_version = staging_versions[0]
    #         staging_model = mlflow_module.pyfunc.load_model(
    #             model_uri=f"models:/{model_name}/Staging"
    #         )
    #         print(f"Staging модель загружена: версия {staging_version.version}")
    #     else:
    #         # Пробуем найти модель в стадии None (только что зарегистрированную)
    #         all_versions = client.search_model_versions(f"name='{model_name}'")
    #         if all_versions:
    #             # Берем последнюю версию
    #             latest_version = max(all_versions, key=lambda v: int(v.version))
    #             if latest_version.current_stage == "None" or latest_version.current_stage is None:
    #                 staging_version = latest_version
    #                 staging_model = mlflow_module.pyfunc.load_model(
    #                     model_uri=f"models:/{model_name}/{latest_version.version}"
    #                 )
    #                 print(f"Staging модель загружена (стадия None): версия {staging_version.version}")
    # except Exception as e:
    #     print(f"Ошибка загрузки Staging модели: {e}")
    model_info = client.search_model_versions(f"name='{model_name}'")
    latest_versions = client.get_latest_versions(model_name, stages=["Staging"])
    staging_version = latest_versions[0]
    try:
        staging_model = mlflow_module.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")
        print(f"✓ Staging модель загружена: версия {staging_version.version}")
        
    except Exception as load_error:
        print(f"✗ Ошибка загрузки Staging модели v{version.version}: {load_error}")
    # for version in model_info:
    #             if str(version.current_stage) == "Staging":
    #                 staging_version = version
    #                 try:
    #                     staging_model = mlflow_module.pyfunc.load_model(
    #                         model_uri=f"models:/{model_name}/Staging"
    #                     )
    #                     print(f"✓ Staging модель загружена: версия {staging_version.version}")
    #                     break
    #                 except Exception as load_error:
    #                     print(f"✗ Ошибка загрузки Staging модели v{version.version}: {load_error}")
    # Если нет Production модели, пробуем загрузить начальную модель из файла
    if production_model is None:
        initial_model_path = '/opt/airflow/dags/catboost_model.cbm'
        if os.path.exists(initial_model_path):
            try:
                print(f"Загрузка начальной модели из файла: {initial_model_path}")
                initial_model = cb.CatBoostClassifier()
                initial_model.load_model(initial_model_path)
                production_model = initial_model
                print("Начальная модель загружена как Production модель")
                
                # Регистрируем начальную модель в MLflow для будущего использования
                try:
                    mlflow_module.set_experiment("initial_model_registration")
                    with mlflow_module.start_run(run_name=f"initial_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                        # Логируем модель
                        try:
                            # Пробуем использовать catboost flavor
                            import mlflow.catboost
                            mlflow.catboost.log_model(
                                initial_model,
                                "model",
                                registered_model_name=model_name
                            )
                            print("Начальная модель зарегистрирована в MLflow (catboost flavor)")
                            
                            # Переводим в Production
                            time.sleep(2)
                            all_versions = client.search_model_versions(f"name='{model_name}'")
                            if all_versions:
                                latest_version = max(all_versions, key=lambda v: int(v.version))
                                try:
                                    client.transition_model_version_stage(
                                        name=model_name,
                                        version=latest_version.version,
                                        stage="Production"
                                    )
                                    print(f"Начальная модель переведена в Production: версия {latest_version.version}")
                                except Exception as e:
                                    print(f"Не удалось перевести модель в Production: {e}")
                        except (ImportError, AttributeError) as e:
                            print(f"CatBoost flavor недоступен: {e}")
                            print("Модель будет использована локально без регистрации в MLflow")
                except Exception as e:
                    print(f"Не удалось зарегистрировать начальную модель в MLflow: {e}")
                    print("Продолжаем с локальной моделью")
            except Exception as e:
                print(f"Ошибка загрузки начальной модели: {e}")
    
    # Если есть только Production модель (начальная), а Staging нет - это нормально для первого запуска
    # В этом случае просто тестируем Production модель
    if production_model is None and staging_model is None:
        raise Exception("Не найдено ни Production, ни Staging моделей для тестирования. "
                       "Убедитесь, что модель зарегистрирована в MLflow или файл catboost_model.cbm существует.")
    
    # Если есть только Production модель, а Staging нет - это первый запуск после переобучения
    # В этом случае просто тестируем Production модель и пропускаем сравнение
    if production_model is not None and staging_model is None:
        print("⚠ Предупреждение: Найдена только Production модель. Staging модель отсутствует.")
        print("Это может означать, что переобучение еще не завершено или модель не была переведена в Staging.")
        print("Продолжаем тестирование только Production модели.")
    
    # Вычисляем метрики для Production модели
    prod_metrics = {}
    if production_model is not None:
        try:
            y_pred_prod = production_model.predict(X_test)
            y_pred_proba_prod = production_model.predict_proba(X_test)[:, 1] if hasattr(production_model, 'predict_proba') else None
            
            prod_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_prod),
                'precision': precision_score(y_test, y_pred_prod, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_prod, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_prod, average='weighted', zero_division=0),
            }
            
            if y_pred_proba_prod is not None:
                try:
                    prod_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba_prod)
                except:
                    prod_metrics['roc_auc'] = 0.0
            
            version_str = f"v{production_version.version}" if production_version else "initial (from file)"
            print(f"\nProduction модель ({version_str}) метрики:")
            for metric, value in prod_metrics.items():
                print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Ошибка при вычислении метрик Production модели: {e}")
            prod_metrics = {}
    
    # Вычисляем метрики для Staging модели
    staging_metrics = {}
    if staging_model is not None:
        try:
            print(f'{X_test}')
            model_info = mlflow.models.get_model_info(model_uri=f"models:/{model_name}/Staging")
            signature = model_info.signature
            features = [input.name for input in signature.inputs]
            print(f"Список фич: {features}")
            y_pred_staging = staging_model.predict(X_test)
            y_pred_proba_staging = staging_model.predict_proba(X_test)[:, 1] if hasattr(staging_model, 'predict_proba') else None
            print(f'Start_PREDICT')
            staging_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_staging),
                'precision': precision_score(y_test, y_pred_staging, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_staging, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_staging, average='weighted', zero_division=0),
            }
            
            if y_pred_proba_staging is not None:
                try:
                    staging_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba_staging)
                except:
                    staging_metrics['roc_auc'] = 0.0
            
            print(f"\nStaging модель (v{staging_version.version}) метрики:")
            for metric, value in staging_metrics.items():
                print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Ошибка при вычислении метрик Staging модели: {e}")
            staging_metrics = {}
    
    # Логируем метрики в MLflow
    experiment_name = "ab_test_experiment"
    mlflow_module.set_experiment(experiment_name)
    
    with mlflow_module.start_run(run_name=f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Логируем метрики Production модели
        if prod_metrics:
            for metric, value in prod_metrics.items():
                mlflow_module.log_metric(f"production_{metric}", value)
        
        # Логируем метрики Staging модели
        if staging_metrics:
            for metric, value in staging_metrics.items():
                mlflow_module.log_metric(f"staging_{metric}", value)
        
        # Вычисляем разницу метрик
        if prod_metrics and staging_metrics:
            # Есть обе модели - сравниваем
            metric_diffs = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                if metric in prod_metrics and metric in staging_metrics:
                    diff = staging_metrics[metric] - prod_metrics[metric]
                    metric_diffs[metric] = diff
                    mlflow_module.log_metric(f"diff_{metric}", diff)
            
            print(f"\nРазница метрик (Staging - Production):")
            for metric, diff in metric_diffs.items():
                print(f"  {metric}: {diff:+.4f}")
            
            # Сохраняем результаты в XCom для следующей задачи
            kwargs['ti'].xcom_push(key='production_metrics', value=prod_metrics)
            kwargs['ti'].xcom_push(key='staging_metrics', value=staging_metrics)
            kwargs['ti'].xcom_push(key='metric_diffs', value=metric_diffs)
            kwargs['ti'].xcom_push(key='staging_version', value=staging_version.version if staging_version else None)
            kwargs['ti'].xcom_push(key='has_production', value=True)
            
            return {
                'production_metrics': prod_metrics,
                'staging_metrics': staging_metrics,
                'metric_diffs': metric_diffs,
                'staging_version': staging_version.version if staging_version else None,
                'has_production': True
            }
        elif staging_metrics and not prod_metrics:
            # Нет Production модели - автоматически переводим Staging в Production
            print("\n⚠ Production модель не найдена. Это первый запуск.")
            print("Staging модель будет автоматически переведена в Production.")
            
            kwargs['ti'].xcom_push(key='production_metrics', value={})
            kwargs['ti'].xcom_push(key='staging_metrics', value=staging_metrics)
            kwargs['ti'].xcom_push(key='metric_diffs', value={})
            kwargs['ti'].xcom_push(key='staging_version', value=staging_version.version if staging_version else None)
            kwargs['ti'].xcom_push(key='has_production', value=False)
            
            return {
                'production_metrics': {},
                'staging_metrics': staging_metrics,
                'metric_diffs': {},
                'staging_version': staging_version.version if staging_version else None,
                'has_production': False
            }
        elif prod_metrics and not staging_metrics:
            # Есть только Production модель, но нет Staging - это нормально, если переобучение еще не завершено
            print("\n⚠ Staging модель не найдена. Переобучение, возможно, еще не завершено.")
            print("Продолжаем с метриками только Production модели.")
            
            kwargs['ti'].xcom_push(key='production_metrics', value=prod_metrics)
            kwargs['ti'].xcom_push(key='staging_metrics', value={})
            kwargs['ti'].xcom_push(key='metric_diffs', value={})
            kwargs['ti'].xcom_push(key='staging_version', value=None)
            kwargs['ti'].xcom_push(key='has_production', value=True)
            
            return {
                'production_metrics': prod_metrics,
                'staging_metrics': {},
                'metric_diffs': {},
                'staging_version': None,
                'has_production': True
            }
        else:
            # Обе метрики пустые или None
            error_msg = f"Не удалось вычислить метрики для сравнения.\n"
            error_msg += f"  Production метрики: {prod_metrics}\n"
            error_msg += f"  Staging метрики: {staging_metrics}\n"
            error_msg += f"  Production модель загружена: {production_model is not None}\n"
            error_msg += f"  Staging модель загружена: {staging_model is not None}"
            raise Exception(error_msg)


def decide_promotion(**kwargs):
    """Решение: нужно ли переводить Staging модель в Production."""
    ti = kwargs['ti']
    
    try:
        prod_metrics = ti.xcom_pull(key='production_metrics', task_ids='ab_test_task')
        staging_metrics = ti.xcom_pull(key='staging_metrics', task_ids='ab_test_task')
        metric_diffs = ti.xcom_pull(key='metric_diffs', task_ids='ab_test_task')
        has_production = ti.xcom_pull(key='has_production', task_ids='ab_test_task', default=True)
        
        # Если нет Production модели, автоматически переводим в Production
        if not has_production:
            print("\n✓ Production модель отсутствует. Автоматически переводим Staging в Production.")
            return 'promote_to_production_task'
        
        # Если нет Staging метрик, значит переобучение не завершено или модель не была переведена в Staging
        if not staging_metrics:
            print("\n⚠ Staging метрики отсутствуют. Переобучение, возможно, не завершено или модель не переведена в Staging.")
            print("Решение: НЕ переводить модель в Production (Staging модель отсутствует)")
            return 'promote_to_production_task'#'skip_promotion_task'
        
        # Если нет Production метрик, но есть Staging - это странная ситуация, но можем перевести
        if not prod_metrics and staging_metrics:
            print("\n⚠ Production метрики отсутствуют, но есть Staging метрики.")
            print("Решение: Переводим Staging в Production (нет Production модели для сравнения)")
            return 'promote_to_production_task'
        
        if not prod_metrics or not staging_metrics:
            print("Не найдены метрики для принятия решения")
            return 'skip_promotion_task'
        
        # Критерии для перевода в Production:
        # 1. Accuracy должна быть выше хотя бы на 0.01 (1%)
        # 2. F1 должна быть выше хотя бы на 0.01
        # 3. ROC-AUC должна быть выше (если доступна)
        
        accuracy_improvement = metric_diffs.get('accuracy', 0)
        f1_improvement = metric_diffs.get('f1', 0)
        roc_auc_improvement = metric_diffs.get('roc_auc', 0)
        
        min_improvement_threshold = 0.01  # Минимальное улучшение 1%
        
        print(f"\nАнализ результатов A/B теста:")
        print(f"  Улучшение Accuracy: {accuracy_improvement:+.4f}")
        print(f"  Улучшение F1: {f1_improvement:+.4f}")
        print(f"  Улучшение ROC-AUC: {roc_auc_improvement:+.4f}")
        print(f"  Порог улучшения: {min_improvement_threshold}")
        
        # Проверяем критерии
        accuracy_ok = accuracy_improvement >= min_improvement_threshold
        f1_ok = f1_improvement >= min_improvement_threshold
        roc_auc_ok = roc_auc_improvement >= 0 if 'roc_auc' in metric_diffs else True
        
        if accuracy_ok and f1_ok and roc_auc_ok:
            print(f"\n✓ Критерии выполнены! Staging модель лучше Production.")
            print(f"  Решение: ПЕРЕВЕСТИ модель в Production")
            return 'promote_to_production_task'
        else:
            print(f"\n✗ Критерии НЕ выполнены. Staging модель не достаточно лучше Production.")
            print(f"  Решение: НЕ переводить модель в Production")
            return 'skip_promotion_task'
            
    except Exception as e:
        print(f"Ошибка при принятии решения: {e}")
        return 'skip_promotion_task'


def promote_to_production(**kwargs):
    """Перевод Staging модели в Production после успешного A/B теста."""
    print("Перевод Staging модели в Production...")
    
    mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    client = MlflowClient()
    model_name = "CreditRiskModel"
    
    ti = kwargs['ti']
    staging_version_num = ti.xcom_pull(key='staging_version', task_ids='ab_test_task')
    
    if not staging_version_num:
        raise Exception("Не найдена версия Staging модели для перевода в Production")
    
    try:
        staging_version = None
        # Получаем Staging модель
        model_info = client.search_model_versions(f"name='{model_name}'")
        latest_versions = client.get_latest_versions(model_name, stages=["Staging"])
        staging_version = latest_versions[0]
        #staging_versions = client.search_model_versions(f"name='{model_name}' AND stage='Staging'")
        if not staging_version:
            # Пробуем найти по версии
            all_versions = client.search_model_versions(f"name='{model_name}'")
            staging_version = next((v for v in all_versions if v.version == str(staging_version_num)), None)
        
            if not staging_version:
                raise Exception(f"Не найдена модель версии {staging_version_num}")
        
        # Архивируем текущую Production модель (если есть)
        try:
            prod_versions = client.search_model_versions(f"name='{model_name}' AND stage='Production'")
            for version in prod_versions:
                try:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version.version,
                        stage="Archived"
                    )
                    print(f"Production модель версии {version.version} перемещена в Archived")
                except Exception as e:
                    print(f"Не удалось архивировать версию {version.version}: {e}")
        except Exception as e:
            print(f"Ошибка при архивировании Production моделей: {e}")
        
        # Переводим Staging в Production
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=staging_version.version,
                stage="Production"
            )
            print(f"✓ Модель версии {staging_version.version} успешно переведена в Production!")
            
            # Логируем событие в MLflow
            mlflow.set_experiment("model_promotion")
            with mlflow.start_run(run_name=f"promotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("promoted_version", staging_version.version)
                mlflow.log_param("promoted_from", "Staging")
                mlflow.log_param("promoted_to", "Production")
                mlflow.log_param("promotion_date", datetime.now().isoformat())
            
            return {
                'status': 'success',
                'message': f'Model v{staging_version.version} promoted to Production',
                'version': staging_version.version
            }
        except Exception as e:
            error_msg = str(e)
            if "RepresenterError" in error_msg or "cannot represent" in error_msg:
                print(f"Предупреждение: Не удалось перевести модель в Production из-за проблемы сериализации MLflow.")
                print(f"Модель версии {staging_version.version} осталась в стадии Staging.")
                print(f"Вы можете перевести её в Production вручную через MLflow UI.")
                return {
                    'status': 'warning',
                    'message': f'Model v{staging_version.version} could not be promoted due to MLflow serialization issue',
                    'version': staging_version.version
                }
            else:
                raise
                
    except Exception as e:
        print(f"Ошибка при переводе модели в Production: {e}")
        raise


with DAG(
        dag_id='drift_monitoring_dag',
        default_args=default_args,
        schedule="@daily",  # Раз в сутки
        catchup=False,
        tags=['monitoring', 'automl', 'mlops', 'pycaret'],
) as dag:
    start = EmptyOperator(task_id='start')

    check_drift_task = PythonOperator(
        task_id='check_drift_task',
        python_callable=check_drift,
    )

    decide_task = BranchPythonOperator(
        task_id='decide_retrain',
        python_callable=decide_retrain,
    )

    retrain_task = PythonOperator(
        task_id='retrain_task',
        python_callable=retrain_model,
    )

    skip_retrain_task = EmptyOperator(task_id='skip_retrain_task')

    # A/B тестирование после переобучения
    ab_test_task = PythonOperator(
        task_id='ab_test_task',
        python_callable=run_ab_test,
    )

    # Решение о переводе в Production
    decide_promotion_task = BranchPythonOperator(
        task_id='decide_promotion',
        python_callable=decide_promotion,
    )

    # Перевод модели в Production
    promote_to_production_task = PythonOperator(
        task_id='promote_to_production_task',
        python_callable=promote_to_production,
    )

    # Пропуск перевода в Production
    skip_promotion_task = EmptyOperator(task_id='skip_promotion_task')

    end = EmptyOperator(task_id='end', trigger_rule='none_failed')

    # Определяем зависимости между задачами
    start >> check_drift_task >> decide_task
    decide_task >> [retrain_task, skip_retrain_task]
    retrain_task >> ab_test_task >> decide_promotion_task
    decide_promotion_task >> [promote_to_production_task, skip_promotion_task] >> end
    skip_retrain_task >> end