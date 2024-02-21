import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import X_test, y_test
import pickle

# Загрузка модели
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
y_pred = loaded_model.predict(X_test)
# Вычисление матрицы ошибок
cm = confusion_matrix(y_test, y_pred)
# Визуализация матрицы ошибок
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt= 'd')
plt.title('Confusion Matrix')
plt.xlabel('Предсказано')
plt.ylabel('Верно')
plt.show()