import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Загружаем данные
data_frames = []
for i in range(5):
    df = pd.read_csv(f'train_data_{i}.csv')
    data_frames.append(df)

# Объединяем все данные в один DataFrame
data = pd.concat(data_frames, ignore_index=True)

# Разделяем данные на признаки (X) и целевую переменную (y)
X = data[['Day']]
y = data['Temperature']

# Создаём и обучаем модель
model = LinearRegression()
model.fit(X, y)

# Сохраняем модель
joblib.dump(model, 'model.pkl')

print("Модель успешно обучена и сохранена!")

