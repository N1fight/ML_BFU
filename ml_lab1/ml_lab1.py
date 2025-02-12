import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Вычисляем статистические данные для списка чисел
def calculate_stats(data):
    return {
        'count': len(data),
        'min': min(data),
        'max': max(data),
        'mean': sum(data) / len(data)
    }

#Читаем данные из CSV файла и возвращаем списки значений для X и Y
def read_data(filename, x_col, y_col):
    x_data, y_data = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= max(x_col, y_col) + 1:
                try:
                    x = float(row[x_col])
                    y = float(row[y_col])
                    x_data.append(x)
                    y_data.append(y)
                except ValueError:
                    continue
    return x_data, y_data

#Вычисляем параметры линейной регрессии
def calculate_regression(x_data, y_data):
    n = len(x_data)
    sum_x = sum(x_data)
    sum_y = sum(y_data)
    sum_x_squared = sum(xi ** 2 for xi in x_data)
    sum_xy = sum(xi * yi for xi, yi in zip(x_data, y_data))

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    return a, b

#Отображаем данные на графике
def plot_data(ax, x_data, y_data, title, xlabel, ylabel, color='red'):
    ax.scatter(x_data, y_data, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

#Отображаем регрессионную прямую на графике
def plot_regression_line(ax, x_data, y_data, a, b):
    x_min, x_max = min(x_data), max(x_data)
    y_pred_min = a * x_min + b
    y_pred_max = a * x_max + b

    ax.scatter(x_data, y_data, color='red')
    ax.plot([x_min, x_max], [y_pred_min, y_pred_max], color='black')
    ax.set_title('Регрессионная прямая')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

#Отображаем квадраты ошибок на графике
def plot_error_squares(ax, x_data, y_data, a, b):
    ax.scatter(x_data, y_data, color='red', zorder=3)
    x_min, x_max = min(x_data), max(x_data)
    y_pred_min = a * x_min + b
    y_pred_max = a * x_max + b
    ax.plot([x_min, x_max], [y_pred_min, y_pred_max], color='black', zorder=2)

    for xi, yi in zip(x_data, y_data):
        y_pred_i = a * xi + b
        error = abs(y_pred_i - yi)

        ax.vlines(xi, min(yi, y_pred_i), max(yi, y_pred_i), color='blue', linestyle='--')

        rect_x = xi - error / 2
        rect_y = min(yi, y_pred_i)

        square = patches.Rectangle((rect_x, rect_y), error, error, facecolor='green', alpha=0.3, edgecolor='black')
        ax.add_patch(square)

    ax.set_title('Квадраты ошибок')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

def main():
    filename = 'student_scores.csv'
    x_col = int(input('Для X введите колонку 0 или 1: '))
    y_col = int(input('Для Y введите колонку 0 или 1: '))

    x_data, y_data = read_data(filename, x_col, y_col)

    stats_x = calculate_stats(x_data)
    stats_y = calculate_stats(y_data)

    print("Статистика для X:")
    print(f"Количество: {stats_x['count']}")
    print(f"Минимум: {stats_x['min']:.2f}")
    print(f"Максимум: {stats_x['max']:.2f}")
    print(f"Среднее: {stats_x['mean']:.2f}")

    print("\nСтатистика для Y:")
    print(f"Количество: {stats_y['count']}")
    print(f"Минимум: {stats_y['min']:.2f}")
    print(f"Максимум: {stats_y['max']:.2f}")
    print(f"Среднее: {stats_y['mean']:.2f}")

    a, b = calculate_regression(x_data, y_data)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    plot_data(ax1, x_data, y_data, 'Исходные данные', 'X', 'Y')
    plot_regression_line(ax2, x_data, y_data, a, b)
    plot_error_squares(ax3, x_data, y_data, a, b)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()