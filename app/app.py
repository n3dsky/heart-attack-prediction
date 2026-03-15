import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import joblib
from catboost import CatBoostClassifier
import io
import uvicorn
from typing import Optional, List, Tuple
import os
from pydantic import BaseModel, Field
import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Уровни риска сердечного приступа"""
    LOW = "Низкий"
    HIGH = "Высокий"


@dataclass
class PredictionResult:
    """Результат предсказания"""
    prediction: int
    probability: float
    risk_level: str


@dataclass
class BatchPredictionResult:
    """Результат пакетного предсказания"""
    predictions: List[int]
    probabilities: List[float]
    statistics: dict
    total_patients: int


class HealthMetrics(BaseModel):
    """Модель для одиночного предсказания"""
    Age: float = Field(..., ge=0, le=120, description="Возраст")
    Cholesterol: float = Field(..., ge=0, description="Уровень холестерина")
    Heart_rate: float = Field(..., ge=0, le=200, description="Частота сердечных сокращений")
    Diabetes: float = Field(..., ge=0, le=1, description="Диабет (0 или 1)")
    Family_History: float = Field(..., ge=0, le=1, description="Семейный анамнез (0 или 1)")
    Smoking: float = Field(..., ge=0, le=1, description="Курение (0 или 1)")
    Obesity: float = Field(..., ge=0, le=1, description="Ожирение (0 или 1)")
    Alcohol_Consumption: float = Field(..., ge=0, le=1, description="Употребление алкоголя (0 или 1)")
    Exercise_Hours_Per_Week: float = Field(..., ge=0, le=168, description="Часов тренировок в неделю")
    Diet: int = Field(..., ge=0, le=1, description="Диета (0 или 1)")
    Previous_Heart_Problems: float = Field(..., ge=0, le=1, description="Предыдущие проблемы с сердцем")
    Medication_Use: float = Field(..., ge=0, le=1, description="Прием лекарств")
    Stress_Level: float = Field(..., ge=0, le=10, description="Уровень стресса")
    Sedentary_Hours_Per_Day: float = Field(..., ge=0, le=24, description="Малоподвижных часов в день")
    Income: float = Field(..., ge=0, description="Доход")
    BMI: float = Field(..., ge=0, le=100, description="Индекс массы тела")
    Triglycerides: float = Field(..., ge=0, description="Триглицериды")
    Physical_Activity_Days_Per_Week: float = Field(..., ge=0, le=7, description="Дней активности в неделю")
    Sleep_Hours_Per_Day: float = Field(..., ge=0, le=24, description="Часов сна в день")
    Blood_sugar: float = Field(..., ge=0, description="Уровень сахара в крови")
    CK_MB: float = Field(..., ge=0, description="Креатинкиназа-MB")
    Troponin: float = Field(..., ge=0, description="Тропонин")
    Gender: str = Field(..., pattern="^(Male|Female)$", description="Пол")
    Systolic_blood_pressure: float = Field(..., ge=0, le=300, description="Систолическое давление")
    Diastolic_blood_pressure: float = Field(..., ge=0, le=200, description="Диастолическое давление")


class PredictionResponse(BaseModel):
    """Модель ответа с предсказаниями"""
    status: str
    predictions: Optional[List[int]] = None
    probabilities: Optional[List[float]] = None
    message: Optional[str] = None
    filename: Optional[str] = None
    statistics: Optional[dict] = None


class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self):
        self.expected_features = [
            'Age', 'Cholesterol', 'Heart rate', 'Diabetes', 'Family History',
            'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',
            'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level',
            'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
            'Physical Activity Days Per Week', 'Sleep Hours Per Day',
            'Blood sugar', 'CK-MB', 'Troponin', 'Gender', 'Systolic blood pressure',
            'Diastolic blood pressure'
        ]
        
        self.categorical_features = [
            'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 
            'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level', 
            'Physical Activity Days Per Week', 'Gender'
        ]
        
        self.column_mapping = {
            'Heart_rate': 'Heart rate',
            'Family_History': 'Family History',
            'Alcohol_Consumption': 'Alcohol Consumption',
            'Exercise_Hours_Per_Week': 'Exercise Hours Per Week',
            'Previous_Heart_Problems': 'Previous Heart Problems',
            'Medication_Use': 'Medication Use',
            'Stress_Level': 'Stress Level',
            'Sedentary_Hours_Per_Day': 'Sedentary Hours Per Day',
            'Physical_Activity_Days_Per_Week': 'Physical Activity Days Per Week',
            'Sleep_Hours_Per_Day': 'Sleep Hours Per Day',
            'Blood_sugar': 'Blood sugar',
            'CK_MB': 'CK-MB',
            'Systolic_blood_pressure': 'Systolic blood pressure',
            'Diastolic_blood_pressure': 'Diastolic blood pressure'
        }
        
        self.prediction_threshold = 0.29
    
    def validate_columns(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Проверка наличия всех необходимых колонок"""
        missing_cols = set(self.expected_features) - set(df.columns)
        extra_cols = set(df.columns) - set(self.expected_features)
        
        if missing_cols:
            return False, f"Отсутствуют колонки: {missing_cols}"
        
        if 'Heart Attack Risk (Binary)' in df.columns:
            logger.info("Найдена целевая переменная 'Heart Attack Risk (Binary)'")
        
        return True, "OK"
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка данных перед предсказанием"""
        df = df.copy()
        
        # Преобразование пола
        df.loc[df['Gender'] == 'Male', 'Gender'] = '1.0'
        df.loc[df['Gender'] == 'Female', 'Gender'] = '0.0'
        df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')
        
        # Заполнение пропусков
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
        
        # Преобразование категориальных признаков
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        return df
    
    def prepare_single_record(self, data: dict) -> pd.DataFrame:
        """Подготовка записи одного пациента"""
        df = pd.DataFrame([data])
        df.rename(columns=self.column_mapping, inplace=True)
        return self.preprocess(df)
    
    def remove_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление целевой переменной если она есть"""
        if 'Heart Attack Risk (Binary)' in df.columns:
            df = df.drop(columns=['Heart Attack Risk (Binary)'])
        return df


class ModelPredictor:
    """Класс для работы с моделью"""
    
    def __init__(self, model_path: str = "catboost_model.cbm"):
        self.model_path = Path(model_path)
        self.model = None
        self.is_loaded = False
        
    def load(self) -> bool:
        """Загрузка модели"""
        try:
            if self.model_path.exists():
                self.model = CatBoostClassifier()
                self.model.load_model(str(self.model_path))
                self.is_loaded = True
                logger.info(f"Модель загружена из {self.model_path}")
                return True
            else:
                logger.warning(f"Файл модели {self.model_path} не найден")
                self.is_loaded = False
                return False
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.is_loaded = False
            return False
    
    async def predict(self, df: pd.DataFrame, threshold: float = 0.29) -> Tuple[List[int], List[float]]:
        """Выполнение предсказаний"""
        if not self.is_loaded or self.model is None:
            return [], []
        
        try:
            # Получение вероятностей
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(df)[:, 1].tolist()
            else:
                probabilities = self.model.predict(df).tolist()
            
            # Применение порога
            predictions = [1 if prob >= threshold else 0 for prob in probabilities]
            
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Ошибка модели: {e}")
            return [], []


class StatisticsCalculator:
    """Класс для расчета статистики"""
    
    @staticmethod
    def calculate_batch_statistics(predictions: List[int], probabilities: List[float]) -> dict:
        """Расчет статистики для пакетных предсказаний"""
        if not predictions:
            return {}
        
        high_risk_count = sum(predictions)
        total = len(predictions)
        
        return {
            "total_patients": total,
            "high_risk_count": int(high_risk_count),
            "low_risk_count": int(total - high_risk_count),
            "high_risk_percentage": float(round(high_risk_count / total * 100, 2)),
            "avg_probability": float(round(sum(probabilities) / total, 4)) if probabilities else 0
        }


class PredictionService:
    """Сервис для обработки предсказаний"""
    
    def __init__(self, model_predictor: ModelPredictor, preprocessor: DataPreprocessor):
        self.model_predictor = model_predictor
        self.preprocessor = preprocessor
        self.statistics_calculator = StatisticsCalculator()
    
    async def predict_from_dataframe(self, df: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """Предсказание на основе DataFrame"""
        # Предобработка
        df_processed = self.preprocessor.preprocess(df)
        df_processed = self.preprocessor.remove_target_column(df_processed)
        
        # Предсказание
        predictions, probabilities = await self.model_predictor.predict(
            df_processed, 
            self.preprocessor.prediction_threshold
        )
        
        return predictions, probabilities
    
    async def predict_single(self, metrics: HealthMetrics) -> PredictionResult:
        """Предсказание для одного пациента"""
        data = metrics.dict()
        df = self.preprocessor.prepare_single_record(data)
        
        predictions, probabilities = await self.predict_from_dataframe(df)
        
        if not predictions:
            raise ValueError("Не удалось выполнить предсказание")
        
        risk_level = RiskLevel.HIGH if predictions[0] == 1 else RiskLevel.LOW
        
        return PredictionResult(
            prediction=predictions[0],
            probability=probabilities[0],
            risk_level=risk_level.value
        )
    
    async def predict_batch(self, df: pd.DataFrame) -> BatchPredictionResult:
        """Пакетное предсказание"""
        # Валидация колонок
        is_valid, message = self.preprocessor.validate_columns(df)
        if not is_valid:
            raise ValueError(message)
        
        predictions, probabilities = await self.predict_from_dataframe(df)
        
        if not predictions:
            raise ValueError("Не удалось выполнить предсказания")
        
        statistics = self.statistics_calculator.calculate_batch_statistics(predictions, probabilities)
        
        return BatchPredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            statistics=statistics,
            total_patients=len(df)
        )
    
    async def predict_from_csv(self, contents: bytes, filename: str) -> BatchPredictionResult:
        """Предсказание из CSV файла"""
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV: {str(e)}")
        
        if df.empty:
            raise ValueError("CSV файл пуст")
        
        return await self.predict_batch(df)


class HeartAttackRiskAPI:
    """Основной класс API"""
    
    def __init__(self, model_path: str = "catboost_model.cbm"):
        self.app = FastAPI(
            title="Heart Attack Risk Prediction API",
            description="API для предсказания риска сердечного приступа на основе медицинских данных",
            version="1.0.0"
        )
        
        # Инициализация компонентов
        self.preprocessor = DataPreprocessor()
        self.model_predictor = ModelPredictor(model_path)
        self.prediction_service = PredictionService(self.model_predictor, self.preprocessor)
        
        # Настройка маршрутов
        self._setup_routes()
    
    def _setup_routes(self):
        """Настройка маршрутов API"""
        
        @self.app.on_event("startup")
        async def startup_event():
            self.model_predictor.load()
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Heart Attack Risk Prediction API",
                "model_loaded": self.model_predictor.is_loaded,
                "features_count": len(self.preprocessor.expected_features),
                "endpoints": {
                    "predict_csv": "/predict/csv (POST) - загрузка CSV файла",
                    "predict_single": "/predict/single (POST) - одиночное предсказание",
                    "predict_batch": "/predict/batch (POST) - пакетное предсказание",
                    "health": "/health (GET)",
                    "features": "/features (GET)"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_loaded": self.model_predictor.is_loaded,
                "service": "Heart Attack Risk Prediction API"
            }
        
        @self.app.get("/features")
        async def get_features():
            """Возвращает список ожидаемых признаков"""
            return {
                "expected_features": self.preprocessor.expected_features,
                "target": "Heart Attack Risk (Binary)",
                "total_features": len(self.preprocessor.expected_features)
            }
        
        @self.app.post("/predict/csv", response_model=PredictionResponse)
        async def predict_from_csv(file: UploadFile = File(...)):
            """
            Предсказание на основе CSV файла с данными пациентов
            """
            try:
                # Проверка формата файла
                if not file.filename.endswith('.csv'):
                    raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
                
                # Чтение и обработка файла
                contents = await file.read()
                result = await self.prediction_service.predict_from_csv(contents, file.filename)
                
                return PredictionResponse(
                    status="success",
                    predictions=result.predictions,
                    probabilities=result.probabilities,
                    filename=file.filename,
                    statistics=result.statistics,
                    message=f"Успешно обработано {result.total_patients} пациентов"
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Ошибка при обработке: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")
        
        @self.app.post("/predict/single")
        async def predict_single(metrics: HealthMetrics):
            """
            Предсказание для одного пациента
            """
            try:
                result = await self.prediction_service.predict_single(metrics)
                
                return {
                    "status": "success",
                    "prediction": result.prediction,
                    "probability": result.probability,
                    "risk_level": result.risk_level,
                    "message": f"Риск сердечного приступа: {result.risk_level} (вероятность: {result.probability:.2%})"
                }
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Ошибка при предсказании: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/batch", response_model=PredictionResponse)
        async def predict_batch(data: List[HealthMetrics]):
            """
            Пакетное предсказание для нескольких пациентов
            """
            try:
                # Преобразование списка объектов в DataFrame
                records = [item.dict() for item in data]
                df = pd.DataFrame(records)
                
                # Переименование колонок
                df.rename(columns=self.preprocessor.column_mapping, inplace=True)
                
                # Выполнение предсказаний
                result = await self.prediction_service.predict_batch(df)
                
                return PredictionResponse(
                    status="success",
                    predictions=result.predictions,
                    probabilities=result.probabilities,
                    statistics=result.statistics,
                    message=f"Успешно обработано {result.total_patients} пациентов"
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Ошибка при пакетном предсказании: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def get_app(self) -> FastAPI:
        """Возвращает экземпляр FastAPI приложения"""
        return self.app


# Создание экземпляра API
api = HeartAttackRiskAPI(model_path="catboost_model.cbm")
app = api.get_app()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )