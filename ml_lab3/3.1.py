import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')

# Визуализация данных Iris с помощью Matplotlib
def plot_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    plt.figure(figsize=(12, 5))

    # График sepal length vs sepal width
    plt.subplot(1, 2, 1)
    for i, name in enumerate(target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=name)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.legend()
    plt.title('Sepal length vs Sepal width')

    # График petal length vs petal width
    plt.subplot(1, 2, 2)
    for i, name in enumerate(target_names):
        plt.scatter(X[y == i, 2], X[y == i, 3], label=name)
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.legend()
    plt.title('Petal length vs Petal width')

    plt.tight_layout()
    plt.show()

    return X, y, iris


# Визуализация с помощью Seaborn pairplot
def plot_pairplot(iris_df):
    sns.pairplot(iris_df, hue='species')
    plt.show()


# Подготовка двух датасетов
def prepare_datasets(X, y):
    # Первый датасет: setosa и versicolor
    mask1 = (y == 0) | (y == 1)
    X1, y1 = X[mask1], y[mask1]

    # Второй датасет: versicolor и virginica
    mask2 = (y == 1) | (y == 2)
    X2, y2 = X[mask2], y[mask2]

    return (X1, y1), (X2, y2)


# Обучение и оценка модели
def train_and_evaluate_model(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = clf.score(X_train, y_train)

    print(f"\nРезультаты для {dataset_name}:")
    print(f"Точность на тестовой выборке: {accuracy:.3f}")
    print(f"Точность на обучающей выборке: {train_accuracy:.3f}")

    return clf


# Генерация синтетического датасета и классификация
def synthetic_dataset_classification():
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                               n_informative=2, random_state=1, n_clusters_per_class=1)

    # Визуализация
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Сгенерированный датасет для бинарной классификации')
    plt.colorbar()
    plt.show()

    # Обучение модели
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = clf.score(X_train, y_train)

    print("\nРезультаты для синтетического датасета:")
    print(f"Точность на тестовой выборке: {accuracy:.3f}")
    print(f"Точность на обучающей выборке: {train_accuracy:.3f}")

    # Визуализация разделяющей поверхности
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Разделяющая поверхность логистической регрессии')
    plt.show()


# Основная программа
def main():
    print("Визуализация данных Iris с помощью Matplotlib")
    X, y, iris = plot_iris_data()

    print("Визуализация с помощью Seaborn pairplot")
    iris_df = pd.DataFrame(X, columns=iris.feature_names)
    iris_df['species'] = [iris.target_names[i] for i in y]
    plot_pairplot(iris_df)

    print("Подготовка двух датасетов:")
    (X1, y1), (X2, y2) = prepare_datasets(X, y)
    print(f"Датасет 1 (setosa vs versicolor): {X1.shape[0]} samples")
    print(f"Датасет 2 (versicolor vs virginica): {X2.shape[0]} samples")

    print("Обучение и оценка моделей для обоих датасетов:")
    model1 = train_and_evaluate_model(X1, y1, "Setosa vs Versicolor")
    model2 = train_and_evaluate_model(X2, y2, "Versicolor vs Virginica")

    print("Генерация синтетического датасета и бинарная классификация:")
    synthetic_dataset_classification()


if __name__ == "__main__":
    main()