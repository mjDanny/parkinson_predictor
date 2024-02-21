import pickle
import matplotlib.pyplot as plt
from model import y_test, X_test

# Загрузка обученной модели
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
y_pred = loaded_model.predict(X_test)
# Вычисление количества правильных и неправильных предсказаний
correct_predictions = (y_pred == y_test).sum()
incorrect_predictions = (y_pred != y_test).sum()

# Создание графика
labels = ['Correct Predictions', 'Incorrect Predictions']
sizes = [correct_predictions, incorrect_predictions]
colors = ['green', 'red']

plt.figure(figsize=(10, 7))
plt.bar(labels, sizes, color=colors)
plt.title('Prediction Results')
plt.ylabel('Number of Predictions')
plt.show()
