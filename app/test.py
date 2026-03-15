import requests
import pandas as pd
import argparse
import json
import random
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Тест health check"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health check:", response.json())
    return response.status_code == 200

def test_predict_csv(csv_path):
    """Тест предсказания из CSV"""
    with open(csv_path, 'rb') as f:
        files = {'file': (Path(csv_path).name, f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/predict/csv", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Предсказания из CSV:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n❌ Ошибка: {response.status_code}")
        print(response.text)

def test_predict_single():
    """Тест одиночного предсказания"""
    patient = {
        "Age": 65,
        "Cholesterol": 240,
        "Heart_rate": 85,
        "Diabetes": 1,
        "Family_History": 1,
        "Smoking": 1,
        "Obesity": 1,
        "Alcohol_Consumption": 0,
        "Exercise_Hours_Per_Week": 2,
        "Diet": 0,
        "Previous_Heart_Problems": 1,
        "Medication_Use": 1,
        "Stress_Level": 7,
        "Sedentary_Hours_Per_Day": 8,
        "Income": 50000,
        "BMI": 28.5,
        "Triglycerides": 180,
        "Physical_Activity_Days_Per_Week": 2,
        "Sleep_Hours_Per_Day": 6,
        "Blood_sugar": 140,
        "CK_MB": 25,
        "Troponin": 0.1,
        "Gender": "Male",
        "Systolic_blood_pressure": 140,
        "Diastolic_blood_pressure": 90
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/single",
        json=patient
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Одиночное предсказание:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n❌ Ошибка: {response.status_code}")
        print(response.text)

def generate_sample_csv(n_patients=10):
    """Генерация тестового CSV"""
    data = []
    
    for i in range(n_patients):
        patient = {
            'Age': random.randint(30, 80),
            'Cholesterol': random.randint(150, 300),
            'Heart rate': random.randint(60, 100),
            'Diabetes': random.choice([0, 1]),
            'Family History': random.choice([0, 1]),
            'Smoking': random.choice([0, 1]),
            'Obesity': random.choice([0, 1]),
            'Alcohol Consumption': random.choice([0, 1]),
            'Exercise Hours Per Week': round(random.uniform(0, 10), 1),
            'Diet': random.choice([0, 1]),
            'Previous Heart Problems': random.choice([0, 1]),
            'Medication Use': random.choice([0, 1]),
            'Stress Level': random.randint(1, 10),
            'Sedentary Hours Per Day': round(random.uniform(4, 12), 1),
            'Income': random.randint(30000, 120000),
            'BMI': round(random.uniform(18.5, 35), 1),
            'Triglycerides': random.randint(100, 250),
            'Physical Activity Days Per Week': random.randint(0, 7),
            'Sleep Hours Per Day': round(random.uniform(5, 9), 1),
            'Blood sugar': random.randint(80, 180),
            'CK-MB': random.randint(5, 30),
            'Troponin': round(random.uniform(0, 0.5), 3),
            'Gender': random.choice(['Male', 'Female']),
            'Systolic blood pressure': random.randint(110, 160),
            'Diastolic blood pressure': random.randint(70, 100)
        }
        data.append(patient)
    
    df = pd.DataFrame(data)
    filename = f'test_patients_{n_patients}.csv'
    df.to_csv(filename, index=False)
    print(f"✅ Сгенерирован файл: {filename}")
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Путь к CSV файлу')
    parser.add_argument('--generate', type=int, help='Сгенерировать тестовый CSV')
    parser.add_argument('--single', action='store_true', help='Тест одиночного предсказания')
    
    args = parser.parse_args()
    
    # Проверка здоровья сервиса
    if not test_health():
        print("❌ Сервис недоступен. Запустите сервер командой: python app.py")
        exit(1)
    
    if args.single:
        test_predict_single()
    elif args.generate:
        csv_file = generate_sample_csv(args.generate)
        test_predict_csv(csv_file)
    elif args.csv:
        test_predict_csv(args.csv)
    else:
        print("Использование:")
        print("  python test_client.py --single              # одиночное предсказание")
        print("  python test_client.py --generate 10         # генерация 10 пациентов")
        print("  python test_client.py --csv data.csv        # предсказание из CSV")