# Часть 2: Многоклассовая логистическая регрессия на наборе  данных Iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('TkAgg')

# Загрузка данных
iris = load_iris()
X = iris.data[:, 2:]  # Берем только petal length и petal width
y = iris.target

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Визуализация
plt.figure(figsize=(10, 6))

# Создание сетки для отрисовки
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Предсказание для каждой точки сетки
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Цветовая карта
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Отрисовка границ решений
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

# Отрисовка обучающих точек
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title(f"Многоклассовая логистическая регрессия на данных Iris")

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()

# Вывод коэффициентов модели
print("\nКоэффициенты модели:")
for i, class_name in enumerate(iris.target_names):
    print(f"\n{class_name}:")
    print(f"  Petal length: {model.coef_[i][0]:.4f}")
    print(f"  Petal width: {model.coef_[i][1]:.4f}")
    print(f"  Intercept: {model.intercept_[i]:.4f}")