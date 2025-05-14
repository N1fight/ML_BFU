# Часть 1: датасет Titanic
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv('Titanic.csv')

# Задание 1: Предобработка данных
print("Исходные размеры данных:", data.shape)

# Удаление строк с пропущенными значениями
data_cleaned = data.dropna()
print("Размер после удаления строк с пропусками:", data_cleaned.shape)

# Удаление нечисловых столбцов, кроме Sex и Embarked
non_numeric_cols = ['Name', 'Ticket', 'Cabin']
data_cleaned = data_cleaned.drop(columns=non_numeric_cols)
print("Столбцы после удаления нечисловых:", data_cleaned.columns.tolist())

# Перекодировка категориальных переменных
data_cleaned['Sex'] = data_cleaned['Sex'].map({'male': 0, 'female': 1})
data_cleaned['Embarked'] = data_cleaned['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

# Удаление PassengerId
data_cleaned = data_cleaned.drop(columns=['PassengerId'])

# Вычисление процента потерянных данных
initial_rows = len(data)
cleaned_rows = len(data_cleaned)
lost_percentage = ((initial_rows - cleaned_rows) / initial_rows) * 100
print(f"\nПроцент потерянных данных: {lost_percentage:.2f}%")

# Задание 2: Машинное обучение
# Разделение данных на обучающую и тестовую выборки
X = data_cleaned.drop(columns=['Survived'])
y = data_cleaned['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели логистической регрессии
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Оценка точности модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели: {accuracy:.4f}")

# Оценка влияния признака Embarked на точность модели
X_no_embarked = X.drop(columns=['Embarked'])
X_train_no, X_test_no, y_train_no, y_test_no = train_test_split(X_no_embarked, y, test_size=0.2, random_state=42)
model_no = LogisticRegression(max_iter=1000, random_state=42)
model_no.fit(X_train_no, y_train_no)
y_pred_no = model_no.predict(X_test_no)
accuracy_no = accuracy_score(y_test_no, y_pred_no)

print(f"Точность модели без признака Embarked: {accuracy_no:.4f}")
print(f"Разница в точности: {accuracy - accuracy_no:.4f}")

# Вывод коэффициентов модели
print("\nКоэффициенты модели:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")