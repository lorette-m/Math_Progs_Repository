import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def quanc8(f, a, b, tol=1e-8, max_depth=30):
    """Адаптивная квадратура Ньютона-Котеса 8-го порядка"""
    # Коэффициенты Ньютона-Котеса для 9 узлов из книги Abramowitz and Stegun. Handbook of Mathematical Functions (1964)
    weights = np.array([989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989]) / 28350

    def integrate_segment(f, a_seg, b_seg):
        """Вычисление интеграла на сегменте [a_seg, b_seg]"""
        nodes = np.linspace(a_seg, b_seg, 9)
        return (b_seg - a_seg) * np.sum(weights * f(nodes))

    stack = [(a, b, 0)]  # (start, end, текущая глубина разбиения)
    integral = 0.0

    while stack:
        a_seg, b_seg, depth = stack.pop()
        if depth > max_depth:
            integral += integrate_segment(f, a_seg, b_seg)
            continue

        # Вычисление интегралов
        whole = integrate_segment(f, a_seg, b_seg)
        mid = (a_seg + b_seg) / 2
        left = integrate_segment(f, a_seg, mid)
        right = integrate_segment(f, mid, b_seg)

        # Проверка точности
        if abs(left + right - whole) < tol:
            integral += left + right
            #print("Достигнутая глубина разделения в QUANC8: ", depth)
        else:
            # Stack - LIFO
            stack.append((mid, b_seg, depth + 1))
            stack.append((a_seg, mid, depth + 1))

    return integral

def solution():
    """Основная функция, вызываемая в main()"""
    # Вычисление значений функции f(x) QUANC8
    x_values = np.arange(3.0, 5.01, 0.25) # [3, 5] h=0.25
    f_values = np.array([quanc8(lambda t: np.cos(t) / t, 1, x, 1e-10) for x in x_values])

    # Полином Лагранжа
    def lagrange_poly(x_nodes, y_nodes, x):
        n = len(x_nodes)
        result = 0.0
        for i in range(n):
            term = y_nodes[i]
            for j in range(n):
                if i != j:
                    term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
            result += term
        return result
    # Построение сплайна
    cs = CubicSpline(x_values, f_values)

    # Точки для сравнения
    x_k = 3.125 + 0.25 * np.arange(8)
    exact_values = np.array([quanc8(lambda t: np.cos(t) / t, 1, x) for x in x_k])

    # Вычисление интерполяций
    spline_vals = cs(x_k)
    poly_vals = np.array([lagrange_poly(x_values, f_values, x) for x in x_k])

    # Создаем плотную сетку для анализа ошибок
    x_fine = np.linspace(3.0, 5.0, 1000)
    exact_fine = np.array([quanc8(lambda t: np.cos(t) / t, 1, x, 1e-12) for x in x_fine])
    spline_fine = cs(x_fine)
    lagrange_fine = np.array([lagrange_poly(x_values, f_values, x) for x in x_fine])

    # Вычисляем отклонения
    spline_error = spline_fine - exact_fine
    lagrange_error = lagrange_fine - exact_fine

    # region Graphics
    # Построение графиков
    plt.figure(figsize=(14, 20))

    # Основной график интерполяций (1/5)
    plt.subplot(5, 1, 1)
    plt.plot(x_fine, exact_fine, 'k-', label='Точное решение', lw=1)
    plt.plot(x_fine, spline_fine, label='Кубический сплайн', alpha=0.8)
    plt.plot(x_fine, lagrange_fine, '--', label='Полином Лагранжа', alpha=0.8)
    plt.scatter(x_values, f_values, color='red', label='Узлы интерполяции')
    plt.title(r'Интерполяция $f(x) = \int_1^x \frac{\cos(t)}{t} dt$')
    plt.legend()
    plt.grid(True)

    # Увеличенный график отклонений (2/5)
    plt.subplot(5, 1, 2)
    zoom_center = 4.25
    x_zoom_half_range = 0.1
    zoom_mask = (x_fine >= zoom_center - x_zoom_half_range) & (x_fine <= zoom_center + x_zoom_half_range)
    x_zoom = x_fine[zoom_mask]

    plt.plot(x_zoom, (spline_fine[zoom_mask] - exact_fine[zoom_mask]) * 1e5, label='Отклонение сплайна ×1e5',
             color='C0')
    plt.plot(x_zoom, (lagrange_fine[zoom_mask] - exact_fine[zoom_mask]) * 1e8, '--', label='Отклонение Лагранжа ×1e8',
             color='C1')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(
        f'Увеличенные отклонения ({zoom_center - x_zoom_half_range:.2f} ≤ x ≤ {zoom_center + x_zoom_half_range:.2f})')
    plt.ylabel('Δf × масштаб')
    plt.grid(True)
    plt.legend()

    # График отклонений сплайна (3/5)
    plt.subplot(5, 1, 3)
    plt.plot(x_fine, spline_error * 1e5, 'b-', label='Отклонение сплайна ×1e5')
    plt.fill_between(x_fine, spline_error * 1e5, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--')

    max_err_x = x_fine[np.argmax(np.abs(spline_error))]
    plt.annotate(f'Макс. отклонение: {np.max(np.abs(spline_error)):.1e}',
                 (max_err_x, np.max(spline_error) * 1e5),
                 arrowprops=dict(arrowstyle="->"),
                 xytext=(max_err_x - 0.3, np.max(spline_error) * 1e5 + 0.5))

    plt.ylim(-1.5, 1.5)
    plt.title('Отклонения кубического сплайна (масштаб 1e-5)')
    plt.ylabel('Δf × 1e5')
    plt.grid(True)

    # График отклонений Лагранжа (4/5)
    plt.subplot(5, 1, 4)
    plt.plot(x_fine, lagrange_error * 1e8, 'r--', label='Отклонение Лагранжа ×1e8')
    plt.fill_between(x_fine, lagrange_error * 1e8, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--')

    plt.ylim(-1.5, 1.5)
    plt.title('Отклонения полинома Лагранжа (масштаб 1e-8)')
    plt.ylabel('Δf × 1e8')
    plt.grid(True)

    # График ошибок с аннотациями (5/5)
    plt.subplot(5, 1, 5)
    plt.semilogy(x_k, np.abs(spline_vals - exact_values), 'o-', label='Ошибка сплайна')
    plt.semilogy(x_k, np.abs(poly_vals - exact_values), 's--', label='Ошибка полинома')

    max_spline_err = np.max(np.abs(spline_vals - exact_values))
    max_poly_err = np.max(np.abs(poly_vals - exact_values))
    plt.annotate(f'Max spline error: {max_spline_err:.1e}',
                 (x_k[np.argmax(np.abs(spline_vals - exact_values))], max_spline_err),
                 textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(f'Max poly error: {max_poly_err:.1e}',
                 (x_k[np.argmax(np.abs(poly_vals - exact_values))], max_poly_err),
                 textcoords="offset points", xytext=(0, -15), ha='center')

    plt.title('Абсолютные ошибки в точках x_k')
    plt.xlabel('x')
    plt.ylabel('|Ошибка|')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # endregion

    # Вывод численных результатов
    print("Точки x_k:\n", x_k)
    print("\nТочные значения:\n", exact_values)
    print("\nЗначения сплайна:\n", spline_vals)
    print("\nОшибки сплайна:\n", np.abs(spline_vals - exact_values))
    print("\nЗначения полинома:\n", poly_vals)
    print("\nОшибки полинома:\n", np.abs(poly_vals - exact_values))