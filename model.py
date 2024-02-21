import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

# считывание датасета
data = pd.read_csv("./data_set/parkinsons.data")
# удаление дубликатов
data = data.drop_duplicates()
# удаление NaN
data = data.dropna()
data = data.dropna(axis=1)
# Заполнение отсутствующих значений предыдущим значением
data = data.fillna(method='ffill')
# Объединение всех признаков в один DataFrame
X = data.drop(['status', 'name'], axis=1)
y = data['status']

# Нормализация признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создание модели XGBoost
model = XGBClassifier()

# Обучение модели
model.fit(X_train, y_train)
# сохранение модели
with open('trained_model.pkl', 'wb')as file:
    pickle.dump(model, file)
# Предсказание меток для тестовой выборки
y_pred = model.predict(X_test)

# Вычисление точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели : {accuracy:.2f}')