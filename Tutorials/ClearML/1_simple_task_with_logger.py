# 1_simple_task_with_logger.py

# Импортируем Task — основная единица эксперимента в ClearML
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from clearml import Task
from clearml import OutputModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Инициализируем задачу
# - project_name: проект в ClearML
# - task_name: имя эксперимента
task = Task.init(project_name="Tutorial", task_name="Simple Task with Logger and Real Training")
task.execute_remotely(queue_name="default")

# Получаем логгер для отправки метрик, графиков, текста и т.д.
logger = task.get_logger()

# Загружаем датасет
print("Загрузка датасета из файла...")
# df = pd.read_csv('Tutorials/ClearML/data/synthetic_dataset.csv')
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# # Предположим, что последняя колонка - это целевая переменная (y), а остальные - признаки (X)
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# Обучаем простую модель
print("Начинаем обучение модели...")
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Обучение с логированием метрик на каждой эпохе (итерации)
n_estimators_range = range(10, 110, 10)  # от 10 до 100 деревьев с шагом 10
train_accuracies = []
val_accuracies = []

for n_est in n_estimators_range:
    model = RandomForestClassifier(n_estimators=n_est, random_state=42)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Подсчет точности
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_test, y_test_pred)
    
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # Логируем скаляр (точность по числу деревьев)
    logger.report_scalar(
        title="Accuracy",
        series="train",
        value=train_acc,
        iteration=n_est
    )
    
    logger.report_scalar(
        title="Accuracy",
        series="validation",
        value=val_acc,
        iteration=n_est
    )
    
    print(f"n_estimators={n_est}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

# Логируем финальную точность
final_accuracy = val_accuracies[-1]
logger.report_single_value(name="final_accuracy", value=final_accuracy)

# Логируем гиперпараметры
hyperparams = {
    'n_estimators': 100,
    'random_state': 42,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
task.connect(hyperparams)  # автоматически логирует гиперпараметры

# Логируем текстовое сообщение
logger.report_text(f"Обучение завершено. Финальная точность: {final_accuracy:.4f}")

# Логируем таблицу с результатами
results_df = pd.DataFrame({
    'n_estimators': n_estimators_range,
    'train_accuracy': train_accuracies,
    'validation_accuracy': val_accuracies
})
logger.report_table(
    title="Training Results",
    series="Results",
    iteration=0,
    table_plot=results_df
)

# Создаем и логируем гистограмму распределения точности
plt.figure(figsize=(10, 6))
plt.hist(train_accuracies, bins=10, alpha=0.5, label='Train Accuracy Distribution', color='blue')
plt.hist(val_accuracies, bins=10, alpha=0.5, label='Validation Accuracy Distribution', color='red')
plt.title('Distribution of Accuracies Across Different Numbers of Estimators')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Логируем matplotlib график
logger.report_matplotlib_figure(
    title="Accuracy Distribution Histogram",
    series="Training Progress",
    iteration=0,
    figure=plt
)

# Создаем и логируем Plotly график Feature Importance
# Обучаем модель с финальным количеством деревьев для получения важности признаков
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

feature_importance = final_model.feature_importances_

fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(range(len(feature_importance))),
    y=feature_importance,
    name='Feature Importance'
))
fig.update_layout(
    title="Feature Importance (Plotly)",
    xaxis_title="Feature Index",
    yaxis_title="Importance"
)

# Логируем Plotly график
logger.report_plotly(
    title="Feature Importance (Plotly)",
    series="Feature Analysis",
    iteration=0,
    figure=fig
)

# Логируем confusion matrix
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

logger.report_confusion_matrix(
    title="Confusion Matrix",
    series="Validation",
    iteration=0,
    matrix=cm,
    xaxis="Predicted",
    yaxis="Actual"
)

# Логируем debug samples (предсказания)
predictions_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred
})
logger.report_table(
    title="Sample Predictions",
    series="Debug Samples",
    iteration=0,
    table_plot=predictions_df.head(20)
)

# Сохраняем и регистрируем модель

# Сохраняем модель во временный файл
model_path = "temp_model.pkl"
joblib.dump(model, model_path, compress=True)

# # Создаем OutputModel и обновляем веса
# output_model = OutputModel(task=task, name="RandomForest Model", tags=['random_forest', 'tutorial'], framework="scikit-learn")
# output_model.update_weights(weights_filename=model_path)

# # Удаляем временный файл
# os.remove(model_path)

# Завершаем задачу (опционально)
task.close()

print(f"✅ Обучение завершено! Финальная точность: {final_accuracy:.4f}")
print("Метрики и графики доступны в веб-интерфейсе ClearML")