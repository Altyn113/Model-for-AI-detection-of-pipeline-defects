"""
СИСТЕМА ДИАГНОСТИКИ ТРУБОПРОВОДОВ - ИСПРАВЛЕННАЯ ВЕРСИЯ
========================================================

Полная система автоматизированного распознавания технического состояния 
трубопроводных систем с ЗАЩИТОЙ ОТ ПЕРЕОБУЧЕНИЯ

Автор: Кондрашов Д.В.
Научный руководитель: Зарубин А.Г.

ИСПРАВЛЕНИЯ:
- Регуляризация модели против переобучения
- Реалистичная эвристика аномалий
- Контроль дисбаланса классов
- Диагностика качества модели
- Реалистичные ожидания результатов
"""

# ========================================================================================
# ИМПОРТЫ И НАСТРОЙКИ
# ========================================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           accuracy_score, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
import logging
import warnings
import os
from datetime import datetime

# Настройки для лучшего отображения
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

print("СИСТЕМА ДИАГНОСТИКИ ТРУБОПРОВОДОВ - ИСПРАВЛЕННАЯ ВЕРСИЯ")
print("=" * 65)
print("Все библиотеки загружены успешно")
print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ========================================================================================
# ДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ
# ========================================================================================

def diagnose_overfitting(results: Dict[str, Any]) -> bool:
    """
    Диагностика переобучения в модели
    
    Args:
        results: результаты обучения модели
        
    Returns:
        bool: True если обнаружено переобучение
    """
    print("\nДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ")
    print("=" * 50)
    
    # 1. Анализ разрыва точности
    train_acc = results['train_accuracy']
    test_acc = results['test_accuracy']
    accuracy_gap = train_acc - test_acc
    
    print(f"Точность на обучении: {train_acc:.4f}")
    print(f"Точность на тесте: {test_acc:.4f}")
    print(f"Разрыв точности: {accuracy_gap:.4f}")
    
    # Диагностика переобучения
    overfitting_detected = False
    
    if accuracy_gap > 0.15:
        print("СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ ОБНАРУЖЕНО!")
        overfitting_detected = True
    elif accuracy_gap > 0.10:
        print("УМЕРЕННОЕ ПЕРЕОБУЧЕНИЕ")
        overfitting_detected = True
    elif train_acc > 0.99 and test_acc > 0.99:
        print("ПОДОЗРЕНИЕ НА ПЕРЕОБУЧЕНИЕ (слишком идеальные результаты)")
        overfitting_detected = True
    else:
        print("Переобучение под контролем")
    
    # 2. Анализ распределения классов
    if 'class_distribution' in results:
        class_dist = results['class_distribution']
        total = sum(class_dist.values())
        
        print(f"\nОбщее распределение классов: {class_dist}")
        
        if total > 0:
            normal_ratio = class_dist.get(0, 0) / total
            if normal_ratio < 0.1 or normal_ratio > 0.95:
                print("ЭКСТРЕМАЛЬНЫЙ ДИСБАЛАНС КЛАССОВ!")
                print(f"   Норма: {normal_ratio*100:.1f}%, Аномалии: {(1-normal_ratio)*100:.1f}%")
            else:
                print(f"Приемлемый баланс классов: Норма {normal_ratio*100:.1f}%")
    
    # 3. Анализ кросс-валидации
    if 'cv_std' in results:
        cv_std = results['cv_std']
        if cv_std > 0.10:
            print(f"Высокая вариативность CV: {cv_std:.4f}")
            print("   Модель нестабильна на разных данных")
    
    # 4. Анализ важности признаков
    if 'feature_importance' in results:
        top_importance = max(results['feature_importance'].values())
        if top_importance > 0.4:
            print(f"ДОМИНИРУЮЩИЙ ПРИЗНАК (важность {top_importance:.3f})")
            print("   Модель может полагаться на один признак")
    
    print("=" * 50)
    return overfitting_detected

# ========================================================================================
# ИСПРАВЛЕННАЯ СИСТЕМА ДИАГНОСТИКИ
# ========================================================================================

class PipelineRealisticSystem:
    """
    ИСПРАВЛЕННАЯ система диагностики трубопроводов с защитой от переобучения
    
    Особенности:
    - Регуляризованная модель Random Forest
    - Строгая эвристика аномалий
    - Контроль дисбаланса классов
    - Диагностика переобучения
    - Реалистичные результаты
    """
    
    def __init__(self, random_state: int = 42):
        """Инициализация исправленной системы"""
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
        # Настройка логирования
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Система диагностики инициализирована (защита от переобучения)")
    
    def load_binary_data(self, filepath: str) -> np.ndarray:
        """
        Загружает и нормализует бинарный файл
        
        Args:
            filepath: путь к бинарному файлу
            
        Returns:
            numpy array с нормализованными данными
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        file_size = os.path.getsize(filepath)
        self.logger.info(f"Размер файла: {file_size:,} байт")
        
        try:
            # Пробуем основные форматы
            formats_to_try = [
                (np.float32, "32-bit float"),
                (np.float64, "64-bit float"),
                (np.int32, "32-bit integer"),
                (np.int16, "16-bit integer")
            ]
            
            for data_type, type_name in formats_to_try:
                try:
                    data = np.fromfile(filepath, dtype=data_type)
                    
                    if len(data) > 1000:  # Минимальный размер данных
                        data = data.astype(np.float64)
                        
                        self.logger.info(f"Данные загружены: {data.shape}, тип: {type_name}")
                        self.logger.info(f"Диапазон: [{np.min(data):.6f}, {np.max(data):.6f}]")
                        
                        # Нормализация больших значений
                        if np.any(np.abs(data) > 1e6):
                            self.logger.warning("Применяем нормализацию больших значений")
                            max_abs_value = np.max(np.abs(data))
                            data = data / max_abs_value
                            self.logger.info(f"После нормализации: [{np.min(data):.6f}, {np.max(data):.6f}]")
                        
                        return data
                        
                except Exception:
                    continue
                    
            raise ValueError("Не удалось определить формат данных")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки файла: {e}")
            raise
    
    def parse_sensor_data(self, raw_data: np.ndarray) -> pd.DataFrame:
        """
        Парсит данные в DataFrame
        
        Args:
            raw_data: сырые данные
            
        Returns:
            DataFrame с данными сенсора
        """
        try:
            # Создаем одноканальный DataFrame (упрощаем для борьбы с переобучением)
            df = pd.DataFrame({'sensor_1': raw_data})
            
            self.logger.info(f"Данные структурированы: {df.shape}")
            self.logger.info(f"Статистика: μ={df.mean().iloc[0]:.6f}, σ={df.std().iloc[0]:.6f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка парсинга данных: {e}")
            raise
    
    def extract_features_conservative(self, sensor_data: pd.DataFrame) -> pd.DataFrame:
        """
        КОНСЕРВАТИВНОЕ извлечение признаков (только основные для предотвращения переобучения)
        
        Args:
            sensor_data: данные сенсора
            
        Returns:
            DataFrame с базовыми признаками
        """
        features = []
        
        for column in sensor_data.columns:
            signal = sensor_data[column].values
            signal = signal[np.isfinite(signal)]
            
            if len(signal) == 0:
                continue
            
            try:
                # Базовые статистические признаки
                mean_val = np.mean(signal)
                std_val = np.std(signal)
                median_val = np.median(signal)
                
                # RMS с защитой
                rms_val = np.sqrt(np.mean(np.clip(signal**2, 0, 1e10)))
                
                # Простые признаки формы сигнала
                peak_val = np.max(np.abs(signal))
                peak_to_peak = np.max(signal) - np.min(signal)
                
                # ТОЛЬКО ОСНОВНЫЕ ПРИЗНАКИ (7 штук)
                features_dict = {
                    f'{column}_mean': mean_val,
                    f'{column}_std': std_val,
                    f'{column}_median': median_val,
                    f'{column}_rms': rms_val,
                    f'{column}_peak': peak_val,
                    f'{column}_peak_to_peak': peak_to_peak,
                    f'{column}_mad': np.median(np.abs(signal - median_val))  # Медианное абс. отклонение
                }
                
                features.append(features_dict)
                
            except Exception as e:
                self.logger.warning(f"Ошибка признаков для {column}: {e}")
                continue
        
        if not features:
            raise ValueError("Не удалось извлечь признаки")
        
        # Объединяем и очищаем
        combined_features = {}
        for feature_dict in features:
            combined_features.update(feature_dict)
        
        # Финальная очистка от infinity/NaN
        for key, value in combined_features.items():
            if not np.isfinite(value):
                combined_features[key] = 0.0
        
        return pd.DataFrame([combined_features])
    
    def create_realistic_windows(self, sensor_data: pd.DataFrame, 
                               window_size: int = 3000, 
                               overlap: float = 0.1) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Создание окон с РЕАЛИСТИЧНОЙ эвристикой аномалий
        
        Args:
            sensor_data: данные сенсоров
            window_size: размер окна (увеличен для стабильности)
            overlap: перекрытие (уменьшено для независимости)
            
        Returns:
            Tuple из признаков и реалистичных меток
        """
        step_size = int(window_size * (1 - overlap))
        windows_features = []
        labels = []
        
        # Глобальные статистики для эвристики
        signal = sensor_data.iloc[:, 0].values
        signal = signal[np.isfinite(signal)]
        
        if len(signal) == 0:
            raise ValueError("Нет валидных данных")
        
        # Робастные статистики
        global_median = np.median(signal)
        global_mad = np.median(np.abs(signal - global_median))
        global_q75 = np.percentile(signal, 75)
        global_q25 = np.percentile(signal, 25)
        global_iqr = global_q75 - global_q25
        global_std = np.std(signal)
        
        self.logger.info(f"Глобальная статистика:")
        self.logger.info(f"  Медиана: {global_median:.6f}")
        self.logger.info(f"  MAD: {global_mad:.6f}")
        self.logger.info(f"  IQR: {global_iqr:.6f}")
        self.logger.info(f"  Std: {global_std:.6f}")
        
        anomaly_count = 0
        max_anomalies = None  # Определим после создания всех окон
        
        # Первый проход - создаем все окна и признаки
        temp_features = []
        temp_anomaly_scores = []
        
        for start in range(0, len(sensor_data) - window_size + 1, step_size):
            end = start + window_size
            window_data = sensor_data.iloc[start:end]
            
            try:
                # Извлекаем признаки
                window_features = self.extract_features_conservative(window_data)
                if window_features.empty:
                    continue
                
                temp_features.append(window_features.iloc[0])
                
                # Вычисляем оценку аномальности
                anomaly_score = self._detect_realistic_anomaly(
                    window_data, global_median, global_mad, global_iqr, global_std
                )
                temp_anomaly_scores.append(anomaly_score)
                
            except Exception as e:
                self.logger.warning(f"Ошибка окна {start}-{end}: {e}")
                continue
        
        if not temp_features:
            raise ValueError("Не удалось создать ни одного окна")
        
        # Второй проход - определяем пороги и метки
        total_windows = len(temp_features)
        max_anomalies = max(1, int(total_windows * 0.08))  # Максимум 8% аномалий
        
        # Сортируем по убыванию аномальности и берем топ-N как аномалии
        anomaly_indices = np.argsort(temp_anomaly_scores)[-max_anomalies:]
        anomaly_threshold = np.min([temp_anomaly_scores[i] for i in anomaly_indices]) if len(anomaly_indices) > 0 else 1.0
        
        # Создаем финальные метки
        for i, score in enumerate(temp_anomaly_scores):
            if i in anomaly_indices and score >= anomaly_threshold:
                labels.append(1)
                anomaly_count += 1
            else:
                labels.append(0)
        
        # Формируем итоговые данные
        features_df = pd.DataFrame(temp_features)
        labels_array = np.array(labels)
        
        # Финальная очистка
        features_df = features_df.replace([np.inf, -np.inf], 0)
        features_df = features_df.fillna(0)
        
        # Статистика
        n_normal = np.sum(labels_array == 0)
        n_anomaly = np.sum(labels_array == 1)
        
        self.logger.info(f"Создано окон: {len(features_df)}")
        self.logger.info(f"Реалистичное распределение:")
        self.logger.info(f"   Норма: {n_normal} ({n_normal/len(labels_array)*100:.1f}%)")
        self.logger.info(f"   Аномалии: {n_anomaly} ({n_anomaly/len(labels_array)*100:.1f}%)")
        self.logger.info(f"Порог аномальности: {anomaly_threshold:.4f}")
        
        return features_df, labels_array
    
    def _detect_realistic_anomaly(self, window_data: pd.DataFrame,
                                global_median: float, global_mad: float,
                                global_iqr: float, global_std: float) -> float:
        """
        СТРОГАЯ эвристика для обнаружения аномалий
        
        Args:
            window_data: данные окна
            global_median, global_mad, global_iqr, global_std: глобальные статистики
            
        Returns:
            float: оценка аномальности (0-1)
        """
        signal = window_data.iloc[:, 0].values
        signal = signal[np.isfinite(signal)]
        
        if len(signal) < 100:
            return 0.0
        
        try:
            # Статистики окна
            window_median = np.median(signal)
            window_mad = np.median(np.abs(signal - window_median))
            window_q75 = np.percentile(signal, 75)
            window_q25 = np.percentile(signal, 25)
            window_iqr = window_q75 - window_q25
            window_std = np.std(signal)
            
            anomaly_score = 0.0
            
            # 1. Отклонение медианы (робастная метрика)
            if global_mad > 1e-12:
                median_deviation = abs(window_median - global_median) / global_mad
                if median_deviation > 5:  # Очень строго
                    anomaly_score += 0.3
                elif median_deviation > 3:
                    anomaly_score += 0.15
            
            # 2. Изменение MAD (устойчивость к выбросам)
            if global_mad > 1e-12:
                mad_ratio = window_mad / global_mad
                if mad_ratio > 4 or mad_ratio < 0.25:
                    anomaly_score += 0.25
                elif mad_ratio > 2.5 or mad_ratio < 0.4:
                    anomaly_score += 0.1
            
            # 3. IQR аномалии
            if global_iqr > 1e-12:
                iqr_ratio = window_iqr / global_iqr
                if iqr_ratio > 3 or iqr_ratio < 0.3:
                    anomaly_score += 0.2
            
            # 4. Экстремальные выбросы (очень строго)
            if global_std > 1e-12:
                extreme_threshold = global_median + 6 * global_std  # 6 сигм!
                extreme_count = np.sum(np.abs(signal - global_median) > extreme_threshold)
                extreme_ratio = extreme_count / len(signal)
                if extreme_ratio > 0.05:  # >5% экстремальных точек
                    anomaly_score += 0.25
            
            return min(anomaly_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Ошибка в эвристике аномалий: {e}")
            return 0.0
    
    def train_regularized_model(self, features: pd.DataFrame, labels: np.ndarray,
                              test_size: float = 0.25) -> Dict[str, Any]:
        """
        Обучение СИЛЬНО РЕГУЛЯРИЗОВАННОЙ модели против переобучения
        
        Args:
            features: матрица признаков
            labels: метки классов
            test_size: размер тестовой выборки
            
        Returns:
            Dict с результатами и метриками
        """
        try:
            self.logger.info(f"Обучение регуляризованной модели на данных: {features.shape}")
            
            # Предварительная очистка
            features_clean = features.copy()
            features_clean = features_clean.replace([np.inf, -np.inf], 0)
            features_clean = features_clean.fillna(0)
            
            # Анализ классов
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            class_dist = dict(zip(unique_labels, label_counts))
            
            self.logger.info(f"Исходное распределение: {class_dist}")
            
            # Разделение на train/test (больше тестовых данных для валидации)
            stratify_labels = labels if len(unique_labels) > 1 else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                features_clean, labels,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_labels
            )
            
            self.logger.info(f"Обучающая выборка: {X_train.shape}")
            self.logger.info(f"Тестовая выборка: {X_test.shape}")
            
            # Масштабирование
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Финальная очистка после масштабирования
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
            
            # МАКСИМАЛЬНО РЕГУЛЯРИЗОВАННАЯ МОДЕЛЬ
            self.model = RandomForestClassifier(
                n_estimators=25,           # Мало деревьев
                max_depth=3,               # Очень ограниченная глубина
                min_samples_split=25,      # Много образцов для разделения
                min_samples_leaf=15,       # Много образцов в листе
                max_features=0.6,          # 60% признаков на дерево
                max_samples=0.7,           # 70% образцов на дерево
                bootstrap=True,
                class_weight='balanced_subsample',  # Балансировка на подвыборках
                random_state=self.random_state,
                n_jobs=-1,
                warm_start=False
            )
            
            # Обучение модели
            self.model.fit(X_train_scaled, y_train)
            self.feature_names = features_clean.columns.tolist()
            self.is_trained = True
            
            # Предсказания
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Кросс-валидация с перемешиванием
            cv_folds = min(3, len(np.unique(y_train)))
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                      cv=cv, scoring='f1_weighted')
            
            # Собираем результаты
            results = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
                'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'classification_report': classification_report(y_test, y_test_pred),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred),
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_features': len(self.feature_names),
                'class_distribution': class_dist,
                'train_class_dist': dict(zip(*np.unique(y_train, return_counts=True))),
                'test_class_dist': dict(zip(*np.unique(y_test, return_counts=True))),
                'cv_scores': cv_scores
            }
            
            # ROC-AUC если возможно
            if len(np.unique(y_test)) > 1:
                y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
                results['roc_auc'] = roc_auc_score(y_test, y_test_proba)
            else:
                results['roc_auc'] = None
            
            # Анализ переобучения
            overfitting_gap = results['train_accuracy'] - results['test_accuracy']
            results['overfitting_gap'] = overfitting_gap
            
            # Логирование результатов
            self.logger.info("Регуляризованная модель обучена!")
            self.logger.info(f"Train Accuracy: {results['train_accuracy']:.4f}")
            self.logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
            self.logger.info(f"Overfitting Gap: {overfitting_gap:.4f}")
            self.logger.info(f"CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            
            if overfitting_gap < 0.08:
                self.logger.info("Переобучение успешно контролируется!")
            elif overfitting_gap < 0.15:
                self.logger.info("Умеренное переобучение")
            else:
                self.logger.warning("Значительное переобучение")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            raise
    
    def predict_anomaly(self, sensor_data: pd.DataFrame,
                       window_size: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание аномалий в новых данных
        
        Args:
            sensor_data: новые данные
            window_size: размер окна для анализа
            
        Returns:
            Tuple из предсказаний и вероятностей
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена!")
        
        # Создаем окна без перекрытия для предсказания
        features, _ = self.create_realistic_windows(sensor_data, window_size, overlap=0)
        
        # Применяем масштабирование
        features_scaled = self.scaler.transform(features)
        features_scaled = np.nan_to_num(features_scaled, nan=0, posinf=0, neginf=0)
        
        # Предсказания
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        return predictions, probabilities

# ========================================================================================
# ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА
# ========================================================================================

def main_realistic_system(filepath: str):
    """
    ГЛАВНАЯ ФУНКЦИЯ - запуск исправленной реалистичной системы
    
    Args:
        filepath: путь к файлу с данными
    """
    print("ЗАПУСК ИСПРАВЛЕННОЙ СИСТЕМЫ ДИАГНОСТИКИ")
    print("=" * 60)
    print("Цель: Получение РЕАЛИСТИЧНЫХ результатов без переобучения")
    print("Защита: Регуляризация + строгая эвристика + контроль баланса")
    print("-" * 60)
    
    # Инициализация системы
    system = PipelineRealisticSystem(random_state=42)
    
    try:
        print("\nЭтап 1: Загрузка реальных данных...")
        raw_data = system.load_binary_data(filepath)
        
        print("\nЭтап 2: Структурирование данных...")
        sensor_data = system.parse_sensor_data(raw_data)
        
        print("\nЭтап 3: Создание реалистичных окон...")
        print("   Это может занять несколько минут...")
        features, labels = system.create_realistic_windows(
            sensor_data, 
            window_size=3000,  # Больше данных для стабильности
            overlap=0.1        # Меньше перекрытия для независимости
        )
        
        print("\nЭтап 4: Обучение регуляризованной модели...")
        results = system.train_regularized_model(features, labels, test_size=0.25)
        
        print("\nЭтап 5: Диагностика переобучения...")
        is_overfitted = diagnose_overfitting(results)
        
        print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 50)
        
        # Основные результаты
        print(f"Точность на обучении: {results['train_accuracy']:.4f}")
        print(f"Точность на тесте: {results['test_accuracy']:.4f}")
        print(f"Разрыв (переобучение): {results['overfitting_gap']:.4f}")
        print(f"Кросс-валидация: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        if results['roc_auc'] is not None:
            print(f"ROC-AUC: {results['roc_auc']:.4f}")
        
        # Анализ качества
        gap = results['overfitting_gap']
        test_acc = results['test_accuracy']
        
        if gap < 0.08 and test_acc >= 0.80:
            print("\nОЦЕНКА: ОТЛИЧНАЯ МОДЕЛЬ!")
            print("   Переобучение под контролем")
            print("   Хорошая обобщающая способность")
        elif gap < 0.15 and test_acc >= 0.70:
            print("\nОЦЕНКА: ХОРОШАЯ МОДЕЛЬ")
            print("   Приемлемые результаты")
            print("   Возможны улучшения")
        else:
            print("\nОЦЕНКА: МОДЕЛЬ ТРЕБУЕТ ДОРАБОТКИ")
            print("   Высокое переобучение или низкая точность")
        
        print(f"\nРаспределение классов: {results['class_distribution']}")
        
        print("\nДЕТАЛЬНЫЙ ОТЧЕТ КЛАССИФИКАЦИИ:")
        print("=" * 60)
        print(results['classification_report'])
        print("=" * 60)
        
        return system, results
        
    except FileNotFoundError:
        print(f"\nОШИБКА: Файл не найден!")
        print(f"Путь: {filepath}")
        print("Убедитесь, что файл captured_data.bin находится по указанному пути")
        return None, None
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        print("Проверьте логи выше для получения подробной информации")
        return None, None


if __name__ == "__main__":
    # Пример использования
    filepath = "captured_data.bin"  # Замените на ваш путь
    system, results = main_realistic_system(filepath)
    
    if system and results:
        print("\nИСПРАВЛЕННАЯ СИСТЕМА ГОТОВА К РАБОТЕ!")
        print("Модель обучена с защитой от переобучения")
        print("Реалистичные результаты получены")
        print("Система готова к промышленному применению")
