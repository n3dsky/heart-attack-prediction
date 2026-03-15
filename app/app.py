import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import joblib
from catboost import CatBoostClassifier
import io
import uvicorn
from typing import Optional, List
import os
from pydantic import BaseModel, Field
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем экземпляр приложения
app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API для предсказания риска сердечного приступа на основе медицинских данных",
    version="1.0.0"
)

# Глобальные переменные
model = None
model_loaded = False
expected_features = [
    'Age', 'Cholesterol', 'Heart rate', 'Diabetes', 'Family History',
    'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',
    'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level',
    'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
    'Physical Activity Days Per Week', 'Sleep Hours Per Day',
    'Blood sugar', 'CK-MB', 'Troponin', 'Gender', 'Systolic blood pressure',
    'Diastolic blood pressure'
]

class PredictionResponse(BaseModel):
    """Модель ответа с предсказаниями"""
    status: str
    predictions: Optional[List[int]] = None
    probabilities: Optional[List[float]] = None
    message: Optional[str] = None
    filename: Optional[str] = None
    statistics: Optional[dict] = None

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

def load_model(model_path: str = "catboost_model.cbm"):
    """Загрузка модели"""
    global model, model_loaded
    try:
        if os.path.exists(model_path):
            model = CatBoostClassifier()
            model.load_model(model_path)
            model_loaded = True
            logger.info(f"Модель загружена из {model_path}")
        else:
            logger.warning(f"Файл модели {model_path} не найден")
            model_loaded = False
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        model_loaded = False

def validate_columns(df: pd.DataFrame) -> tuple:
    """Проверка наличия всех необходимых колонок"""
    missing_cols = set(expected_features) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_features)
    
    if missing_cols:
        return False, f"Отсутствуют колонки: {missing_cols}"
    
    # Проверка наличия целевой переменной (опционально)
    if 'Heart Attack Risk (Binary)' in df.columns:
        logger.info("Найдена целевая переменная 'Heart Attack Risk (Binary)'")
    
    return True, "OK"

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Предобработка данных перед предсказанием"""
    df = df.copy()
    
    df.loc[df['Gender'] == 'Male', 'Gender'] = '1.0'
    df.loc[df['Gender'] == 'Female', 'Gender'] = '0.0'
    df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        elif df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
    
    cat_features = ['Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Diet',
                    'Previous Heart Problems', 'Medication Use', 'Stress Level', 
                    'Physical Activity Days Per Week', 'Gender']
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {
        "message": "Heart Attack Risk Prediction API",
        "model_loaded": model_loaded,
        "features_count": len(expected_features),
        "endpoints": {
            "predict_csv": "/predict/csv (POST) - загрузка CSV файла",
            "predict_single": "/predict/single (POST) - одиночное предсказание",
            "predict_batch": "/predict/batch (POST) - пакетное предсказание",
            "health": "/health (GET)",
            "features": "/features (GET)"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "service": "Heart Attack Risk Prediction API"
    }

@app.get("/features")
async def get_features():
    """Возвращает список ожидаемых признаков"""
    return {
        "expected_features": expected_features,
        "target": "Heart Attack Risk (Binary)",
        "total_features": len(expected_features)
    }

@app.post("/predict/csv", response_model=PredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Предсказание на основе CSV файла с данными пациентов
    """
    try:
        # Проверка формата файла
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
        
        # Чтение файла
        contents = await file.read()
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {str(e)}")
        
        # Проверка данных
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV файл пуст")
        
        # Валидация колонок
        is_valid, message = validate_columns(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Предобработка
        df_processed = preprocess_data(df)
        
        # Удаление целевой переменной если есть
        if 'Heart Attack Risk (Binary)' in df_processed.columns:
            df_processed = df_processed.drop(columns=['Heart Attack Risk (Binary)'])
        
        # Предсказание
        predictions, probabilities = await make_predictions(df_processed)
        
        # Статистика
        stats = {
            "total_patients": len(df),
            "high_risk_count": int(sum(predictions)),
            "low_risk_count": int(len(predictions) - sum(predictions)),
            "high_risk_percentage": float(round(sum(predictions) / len(predictions) * 100, 2))
        }
        
        return PredictionResponse(
            status="success",
            predictions=predictions,
            probabilities=probabilities,
            filename=file.filename,
            statistics=stats,
            message=f"Успешно обработано {len(df)} пациентов"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Ошибка при обработке: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")

@app.post("/predict/single")
async def predict_single(metrics: HealthMetrics):
    """
    Предсказание для одного пациента
    """
    try:
        # Преобразование в DataFrame
        data = metrics.dict()
        
        # Переименование полей для соответствия датасету
        rename_map = {
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
        
        df = pd.DataFrame([data])
        df.rename(columns=rename_map, inplace=True)
        
        # Предобработка
        df_processed = preprocess_data(df)
        
        # Предсказание
        predictions, probabilities = await make_predictions(df_processed)
        
        risk_level = "Высокий" if predictions[0] == 1 else "Низкий"
        
        return {
            "status": "success",
            "prediction": predictions[0],
            "probability": probabilities[0],
            "risk_level": risk_level,
            "message": f"Риск сердечного приступа: {risk_level} (вероятность: {probabilities[0]:.2%})"
        }
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def make_predictions(df: pd.DataFrame) -> tuple:
    """Выполнение предсказаний"""
    global model, model_loaded
    
    if model_loaded and model is not None:
        try:
            # Предсказание классов
            predictions = model.predict(df).tolist()
            
            # Вероятности (если модель поддерживает)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df)[:, 1].tolist()
            else:
                probabilities = [float(p) for p in predictions]
                
            threshold = 0.29
            predictions = [1 if prob >= threshold else 0 for prob in probabilities]
            
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Ошибка модели: {e}")
            return []
    else:
        return []

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )