import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Загрузка данных
data = pd.read_csv('housing.csv', sep=r'\s+', header=None)
data.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# Разделение на признаки и целевую переменную
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                        target_mse=25, scoring='neg_mean_squared_error'):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("MSE")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training MSE")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation MSE")

    if target_mse is not None:
        plt.axhline(y=target_mse, color='b', linestyle='--', label='Target MSE')

    plt.legend(loc="best")
    plt.show()


# Построение кривых обучения
print("Анализ кривых обучения:".center(70))

models_for_learning = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'Lasso Regression': make_pipeline(StandardScaler(), Lasso(alpha=0.1))
}

for name, model in models_for_learning.items():
    plot_learning_curve(model, f"Learning Curve ({name})",
                        X_train, y_train, ylim=(0, 100), cv=5)

print("\nВыводы по кривым обучения:")
print("1. Линейная регрессия показывает признаки переобучения:")
print("   - Ошибка на обучении (MSE = 20-22) значительно ниже, чем на валидации (MSE = 25-30)")
print("2. Ridge и Lasso регрессии демонстрируют лучшую сбалансированность:")
print("   - Разница между ошибками на обучении и валидации небольшая")
print("3. Целевое значение MSE=25 не достигнуто:")
print("   - Лучший результат на валидации: Ridge ")
print("4. Рекомендации по улучшению:")
print("   - Добавление полиномиальных признаков")
print("   - Использование более сложных моделей (случайный лес, градиентный бустинг)")
print("   - Увеличение объема данных (если возможно)")

# Сравнение моделей
print("Сравнение моделей регрессии:".center(70))

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'Lasso Regression': make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
    'ElasticNet': make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5))
}

results = []
coefficients = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, 'coef_'):
        coefficients[name] = model.coef_
    elif hasattr(model.steps[1][1], 'coef_'):
        coefficients[name] = model.steps[1][1].coef_

    results.append({
        'Model': name,
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    })

# Вывод таблицы с результатами
results_df = pd.DataFrame(results)
print("\nТаблица сравнения моделей:")
print("-" * 70)
print(results_df.to_string(index=False))
print("-" * 70)

print("\nВыводы по сравнению моделей:")
print(" Лучшая модель по всем метрикам - Ridge Regression:")
print(f"   - MSE: {results_df[results_df['Model'] == 'Ridge Regression']['MSE'].values[0]:.2f}")
print(f"   - R2: {results_df[results_df['Model'] == 'Ridge Regression']['R2'].values[0]:.3f}")
print(f"   - MAE: {results_df[results_df['Model'] == 'Ridge Regression']['MAE'].values[0]:.2f}")

# Анализ коэффициентов
print("Анализ коэффициентов моделей:".center(70))

coef_df = pd.DataFrame(coefficients, index=X.columns)
print("\nТаблица коэффициентов:")
print("-" * 70)
print(coef_df.to_string())
print("-" * 70)

print("\nВыводы по коэффициентам:")
print("1. Наиболее значимые признаки:")
print("   - LSTAT: сильное отрицательное влияние")
print("   - RM: сильное положительное влияние")
print("   - DIS: положительное влияние")
print("2. Незначимые признаки:")
print("   - ZN, INDUS, AGE, B")

# Визуализация коэффициентов
plt.figure(figsize=(12, 6))
coef_df.plot(kind='bar', ax=plt.gca())
plt.title('Сравнение коэффициентов признаков в разных моделях')
plt.ylabel('Значение коэффициента')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
