import requests
import pandas as pd
import argparse
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_predict_csv(csv_path):
    """Тест предсказания из CSV"""
    with open(csv_path, 'rb') as f:
        files = {'file': (Path(csv_path).name, f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/predict/csv", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Предсказания из CSV:")
        print(result['predictions'], len(result['predictions']))
        return result['predictions']
        
    else:
        print(f"\n❌ Ошибка: {response.status_code}")
        print(response.text)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Путь к CSV файлу для тестовых предсказаний")
    args = parser.parse_args()
    
    if args.csv:
        df = pd.read_csv(args.csv)
        predictions = test_predict_csv(args.csv)
        predictions_df = pd.DataFrame(df['id'], columns=['id'])
        predictions_df['prediction'] = predictions
        print(predictions_df.head())
        predictions_df.to_csv("test_predictions.csv", sep=";", encoding="utf-8", na_rep="MISSING", index=False)