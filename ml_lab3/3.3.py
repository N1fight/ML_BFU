import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, roc_curve, auc,
                             classification_report)
from sklearn.preprocessing import label_binarize


class TitanicModelEvaluator:
    def __init__(self, data_path='titanic.csv'):
        self.data_path = data_path
        self.models = {}
        self.metrics = {}
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """Загрузка и предварительная обработка данных"""
        # Загрузка данных
        self.titanic = pd.read_csv(self.data_path)

        # Очистка данных
        self.titanic_clean = self.titanic.dropna()
        cols_to_drop = [col for col in self.titanic_clean.columns
                        if self.titanic_clean[col].dtype == 'object' and col not in ['Sex', 'Embarked']]
        self.titanic_clean = self.titanic_clean.drop(cols_to_drop, axis=1)

        # Преобразование категориальных признаков
        self.titanic_clean['Sex'] = self.titanic_clean['Sex'].map({'male': 0, 'female': 1})
        self.titanic_clean['Embarked'] = self.titanic_clean['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
        self.titanic_clean = self.titanic_clean.drop('PassengerId', axis=1)

        # Разделение на признаки и целевую переменную
        self.X = self.titanic_clean.drop('Survived', axis=1)
        self.y = self.titanic_clean['Survived']

        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

    def train_models(self):
        """Обучение всех моделей"""
        self._train_logistic_regression()
        self._train_svm()
        self._train_knn()

    def _train_logistic_regression(self):
        """Обучение модели логистической регрессии"""
        print("Часть 1: Логистическая регрессия с расширенными метриками")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        self.metrics['Logistic Regression'] = self._evaluate_model(model, "Logistic Regression")

    def _train_svm(self):
        """Обучение SVM модели"""
        model = SVC(probability=True, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['SVM'] = model
        self.metrics['SVM'] = self._evaluate_model(model, "SVM")

    def _train_knn(self):
        """Обучение KNN модели"""
        model = KNeighborsClassifier()
        model.fit(self.X_train, self.y_train)
        self.models['KNN'] = model
        self.metrics['KNN'] = self._evaluate_model(model, "KNN")

    def _evaluate_model(self, model, model_name):
        """Оценка модели и визуализация метрик"""
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(self.y_test)

        # Вывод отчета о классификации с zero_division=0
        print(f"\n=== {model_name} ===")
        print(classification_report(self.y_test, y_pred, zero_division=0))

        # Визуализация матрицы ошибок
        self._plot_confusion_matrix(y_pred, model_name)

        # Визуализация PR-кривой
        self._plot_precision_recall_curve(y_proba, model_name)

        # Визуализация ROC-кривой
        roc_auc = self._plot_roc_curve(y_proba, model_name)

        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc
        }

    def _plot_confusion_matrix(self, y_pred, model_name):
        """Визуализация матрицы ошибок"""
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Survived', 'Survived'],
                    yticklabels=['Not Survived', 'Survived'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        plt.close()

    def _plot_precision_recall_curve(self, y_proba, model_name):
        """Визуализация PR-кривой"""
        plt.figure(figsize=(6, 4))
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        plt.plot(recall, precision, marker='.')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.tight_layout()
        plt.show()
        plt.close()

    def _plot_roc_curve(self, y_proba, model_name):
        """Визуализация ROC-кривой и вычисление AUC"""
        plt.figure(figsize=(6, 4))
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve - {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
        return roc_auc

    def compare_models(self):
        """Сравнение метрик всех моделей"""
        print("Часть 2: Сравнение моделей классификации")

        # Создание DataFrame с метриками
        metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index')
        print("\nСравнение метрик всех моделей:")
        print(metrics_df)

        # Визуализация сравнения метрик
        self._plot_metrics_comparison(metrics_df)

        # Вывод наилучшей модели
        best_model = metrics_df['f1'].idxmax()
        print(f"\nНаилучшая модель по F1-score: {best_model}")

    def _plot_metrics_comparison(self, metrics_df):
        """Визуализация сравнения метрик моделей"""
        plt.figure(figsize=(10, 5))
        metrics_df.plot(kind='bar', rot=0)
        plt.title('Сравнение метрик моделей')
        plt.ylabel('Значение метрики')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        plt.close()


# Основной блок выполнения
if __name__ == "__main__":
    evaluator = TitanicModelEvaluator()
    evaluator.train_models()
    evaluator.compare_models()