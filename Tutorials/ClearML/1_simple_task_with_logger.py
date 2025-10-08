# 1_simple_task_with_logger.py

# Импортируем библиотеки для визуализации
import matplotlib.pyplot as plt  # библиотека для создания статических графиков
import pandas as pd             # библиотека для работы с табличными данными
import plotly.graph_objects as go  # библиотека для интерактивной визуализации
import joblib                   # библиотека для сохранения и загрузки моделей
import os                       # библиотека для работы с операционными системами

# Импортируем основные компоненты ClearML
from clearml import Task        # основной класс для создания экспериментов в ClearML

# Импортируем компоненты scikit-learn для машинного обучения
from sklearn.preprocessing import PolynomialFeatures  # класс для создания полиномиальных признаков
from sklearn.linear_model import LogisticRegression   # логистическая регрессия для бинарной классификации
from sklearn.pipeline import Pipeline                 # конвейер для объединения трансформеров и модели
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc # метрики для оценки качества модели
from sklearn.model_selection import train_test_split          # функция для разделения данных
from sklearn.decomposition import PCA                 # анализ главных компонент

# Инициализируем задачу в ClearML
# - project_name: имя проекта в ClearML, в котором будет зарегистрирован эксперимент
# - task_name: уникальное имя эксперимента
# - output_uri: позволяет сохранять результаты эксперимента в удалённом хранилище
task: Task = Task.init(
    project_name="Tutorial",                    # имя проекта в ClearML
    task_name="Simple Task with Polynomial Regression",  # имя эксперимента
    output_uri=True,                           # включаем сохранение результатов в удалённое хранилище
)
# Добавляем теги к задаче
task.add_tags(["polynomial-regression", "tutorial"])
# Получаем логгер для отправки метрик, графиков, текста и других артефактов в ClearML
logger = task.get_logger()

# Загружаем датасет
print("Загрузка датасета из файла...")  # выводим сообщение о начале загрузки данных
df = pd.read_csv("Tutorials/ClearML/data/synthetic_dataset.csv")  # загружаем табличные данные из CSV файла

# Разделяем признаки (X) и целевую переменную (y)
# Предположим, что последняя колонка - это целевая переменная (y), а остальные - признаки (X)
X = df.drop("target", axis=1)  # X содержит все признаки (все колонки кроме 'target')
y = df["target"]               # y содержит целевую переменную (колонка 'target')

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                    # признаки и целевая переменная
    test_size=0.2,           # 20% данных выделяется на тестовую выборку
    random_state=42          # фиксированное значение для воспроизводимости результата
)

# Выводим информацию о размерах выборок
print(f"Размер обучающей выборки: {X_train.shape}")  # количество строк и столбцов в обучающей выборке
print(f"Размер тестовой выборки: {X_test.shape}")    # количество строк и столбцов в тестовой выборке

# Создаем и логируем matplotlib график PCA scatter plot
print("Создаем и логируем matplotlib график PCA scatter plot")
pca = PCA(n_components=2)  # уменьшаем размерность до 2 компонент
X_pca = pca.fit_transform(X) # преобразуем исходные данные

# Создаем scatter plot с цветом точек в зависимости от класса
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)  # добавляем цветовую шкалу
plt.title("PCA Scatter Plot (2 Components)")  # заголовок графика
plt.xlabel("First Principal Component")  # подпись оси X
plt.ylabel("Second Principal Component")  # подпись оси Y

# Логируем matplotlib график
logger.report_matplotlib_figure(
    title="Dataset Visualization",  # заголовок графика в ClearML
    series="PCA Scatter Plot",     # серия данных
    figure=plt,                    # сам объект matplotlib фигуры
)

# Определяем гиперпараметры для модели
# Эти параметры будут использоваться в процессе обучения и оптимизации
hyperparams = {
    # Диапазон степеней полинома для тестирования (от 1 до 5)
    "poly_degree_range": list(range(1, 5)),
    # Фиксированное значение для воспроизводимости результатов
    "random_state": 2,
    # Параметр регуляризации для логистической регрессии (обратно пропорционально силе регуляризации)
    "C": 1.0,
    # Максимальное количество итераций для сходимости
    "max_iter": 100,
}
# Подключаем гиперпараметры к задаче для автоматического логирования
task.connect(hyperparams)

# Обучаем простую модель
print("Начинаем обучение модели...")

# Обучение с логированием метрик на каждой эпохе (итерации)
# Используем гиперпараметры, определенные ранее
poly_degree_range = hyperparams["poly_degree_range"]  # диапазон степеней полинома для тестирования
train_accuracies = []  # список для хранения точностей на обучающей выборке
val_accuracies = []    # список для хранения точностей на валидационной выборке

# Проходим по каждому значению степени полинома из диапазона
for degree in poly_degree_range:
    # Создаем модель с текущими гиперпараметрами
    # Используем Pipeline для объединения PolynomialFeatures и LogisticRegression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),  # преобразование признаков в полиномиальные
        ('logistic', LogisticRegression(              # логистическая регрессия для бинарной классификации
            random_state=hyperparams["random_state"], # фиксированное значение для воспроизводимости
            C=hyperparams["C"],                       # параметр регуляризации
            max_iter=hyperparams["max_iter"],         # максимальное количество итераций для сходимости
        ))
    ])
    # Обучаем модель на обучающей выборке
    model.fit(X_train, y_train)

    # Получаем предсказания модели на обучающей и тестовой выборках
    y_train_pred = model.predict(X_train)  # предсказания на обучающей выборке
    y_test_pred = model.predict(X_test)    # предсказания на тестовой выборке

    # Вычисляем точность модели на обучающей и тестовой выборках
    train_acc = accuracy_score(y_train, y_train_pred)  # точность на обучающей выборке
    val_acc = accuracy_score(y_test, y_test_pred)      # точность на тестовой выборке

    # Добавляем полученные точности в соответствующие списки
    train_accuracies.append(train_acc)  # добавляем точность на обучающей выборке
    val_accuracies.append(val_acc)      # добавляем точность на тестовой выборке

    # Логируем скалярные значения точности для визуализации в ClearML
    logger.report_scalar(
        title="Accuracy",      # заголовок графика
        series="train",        # серия данных (обучающая выборка)
        value=float(train_acc), # значение точности
        iteration=degree,      # итерация (используется как ось X на графике)
    )

    logger.report_scalar(
        title="Accuracy",      # заголовок графика
        series="validation",   # серия данных (валидационная/тестовая выборка)
        value=float(val_acc),  # значение точности
        iteration=degree,      # итерация (используется как ось X на графике)
    )

    # Выводим информацию о текущем прогрессе обучения в консоль
    print(
        f"polynomial_degree={degree}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
    )

# Находим лучшую точность на валидационной выборке и соответствующее значение степени полинома
best_val_accuracy = max(val_accuracies)  # наилучшая точность среди всех значений
best_poly_degree = poly_degree_range[val_accuracies.index(best_val_accuracy)]  # значение степени полинома, соответствующее лучшей точности

# Логируем финальную точность
final_accuracy = best_val_accuracy
logger.report_single_value(name="final_accuracy", value=final_accuracy)

# Логируем лучшие гиперпараметры
best_hyperparams = {
    "best_poly_degree": best_poly_degree,
    "random_state": hyperparams["random_state"],
    "C": hyperparams["C"],
    "max_iter": hyperparams["max_iter"],
}

logger.report_text(f"Лучшие гиперпараметры: {best_hyperparams}")

# Логируем текстовое сообщение (по сути тоже что и print, но с возможностью не выводить в консоль или добавить уровень логирования)
logger.report_text(
    f"Обучение завершено. Финальная точность: {final_accuracy:.4f}"
)

# Логируем таблицу с результатами
# Создаем DataFrame с результатами для удобного отображения
print("Создаем DataFrame с результатами и logger.report_table")
results_df = pd.DataFrame(
    {
        "polynomial_degree": poly_degree_range,    # степень полинома
        "train_accuracy": train_accuracies,        # точность на обучающей выборке
        "validation_accuracy": val_accuracies,     # точность на валидационной выборке
    }
)
logger.report_table(
    title="Training Results",     # заголовок таблицы в ClearML
    series="Results",             # серия данных
    iteration=0,                 # итерация (для согласованности)
    table_plot=results_df,       # сам DataFrame с результатами
)

# Создаем и логируем Plotly график ROC-кривой для финальной модели
# Обучаем финальную модель с лучшими гиперпараметрами
print("Обучение финальной модели...")
final_model = Pipeline([
    ('poly', PolynomialFeatures(degree=best_poly_degree)),  # преобразование признаков в полиномиальные
    ('logistic', LogisticRegression(                        # логистическая регрессия для бинарной классификации
        random_state=hyperparams["random_state"],          # фиксированное значение для воспроизводимости
        C=hyperparams["C"],                                # параметр регуляризации
        max_iter=hyperparams["max_iter"],                  # максимальное количество итераций для сходимости
    ))
])
final_model.fit(X_train, y_train)  # обучаем модель с лучшими гиперпараметрами

# Получаем вероятности предсказаний для построения ROC-кривой
y_pred_proba = final_model.predict_proba(X_test)[:, 1]  # вероятности для положительного класса
y_pred = final_model.predict(X_test) # получаем предсказания на тестовой выборке

# Вычисляем значения для ROC-кривой
print("Вычисляем ROC curve и строим через plotly")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)  # ложноположительные и истинноположительные rates
roc_auc = auc(fpr, tpr)  # площадь под ROC-кривой

fig = go.Figure()  # создаем объект графика Plotly
# Добавляем ROC-кривую
fig.add_trace(
    go.Scatter(
        x=fpr,                   # ось X: ложноположительные rates
        y=tpr,                   # ось Y: истинноположительные rates
        mode='lines',            # режим отображения: линии
        name=f'ROC Curve (AUC = {roc_auc:.4f})',  # название серии данных с AUC
    )
)
# Добавляем диагональную линию (random classifier)
fig.add_trace(
    go.Scatter(
        x=[0, 1],               # ось X: от 0 до 1
        y=[0, 1],               # ось Y: от 0 до 1
        mode='lines',           # режим отображения: линии
        name='Random Classifier',  # название случайного классификатора
        line=dict(dash='dash'), # пунктирная линия
    )
)
fig.update_layout(
    title="ROC Curve (Plotly)",  # заголовок графика
    xaxis_title="False Positive Rate",  # подпись оси X
    yaxis_title="True Positive Rate",   # подпись оси Y
    xaxis=dict(range=[0, 1]),           # диапазон оси X
    yaxis=dict(range=[0, 1]),           # диапазон оси Y
)

# Логируем Plotly график
logger.report_plotly(
    title="Training Results", # заголовок графика в ClearML
    series="ROC Curve",         # серия данных
    figure=fig,                 # сам объект графика
)

# Логируем confusion matrix
print("Вычисляем confusion matrix...")
cm = confusion_matrix(y_test, y_pred)  # вычисляем матрицу ошибок

logger.report_confusion_matrix(
    title="Confusion Matrix",  # заголовок матрицы ошибок в ClearML
    series="Validation",       # серия данных
    iteration=0,              # итерация (для согласованности)
    matrix=cm,                # сама матрица ошибок
    xaxis="Predicted",        # подпись оси X
    yaxis="Actual",           # подпись оси Y
)

# Логируем примеры предсказаний для отладки
print("Логгируем часть предсказаний")
predictions_df = pd.DataFrame(
    {"true_label": y_test, "predicted_label": y_pred}  # создаем DataFrame с истинными и предсказанными метками
)
logger.report_table(
    title="Sample Predictions",      # заголовок таблицы в ClearML
    series="Debug Samples",          # серия данных
    iteration=0,                    # итерация (для согласованности)
    table_plot=predictions_df.head(20),  # первые 20 строк таблицы с предсказаниями
)

# Сохраняем и регистрируем модель

# Сохраняем финальную модель в файл
model_path = "models/polynomial.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(final_model, model_path, compress=True)  # сохраняем финальную модель с лучшими гиперпараметрами

# Завершаем задачу (опционально)
task.close()  # закрываем задачу в ClearML, чтобы указать, что эксперимент завершен
print(f"✅ Обучение завершено! Финальная точность: {final_accuracy:.4f}")  # выводим итоговую точность
print("Метрики и графики доступны в веб-интерфейсе ClearML")  # информируем пользователя о доступности результатов
