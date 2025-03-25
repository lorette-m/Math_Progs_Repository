import numpy as np
from scipy.integrate import solve_ivp

# Определение системы дифф уров
def system(t, y):
    x1, x2 = y
    dx1dt = -310 * x1 - 3000 * x2 + 1 / (10 * t ** 2 + 1) if t != 0 else -310 * x1 - 3000 * x2 + 1
    dx2dt = x1 + np.exp(-2 * t)
    return [dx1dt, dx2dt]

# Начальные условия
y0 = [0.0, 1.0]  # x1, x2
t_span = [0, 0.4]
h_print = 0.02 # Шаг для вывода
t_eval = np.arange(0, 0.4 + h_print, h_print)

# Метод III: RKF45
sol_rkf45 = solve_ivp(system, t_span, y0, method='RK45', t_eval=t_eval, rtol=1e-4, atol=1e-4)

# Метод IVa и IVб: Рунге-Кутты 4-го порядка
def runge_kutta(f, t, y, h):
    k1 = h * np.array(f(t, y))
    k2 = h * np.array(f(t + h / 3, y + k1 / 3))
    k3 = h * np.array(f(t + 2 * h / 3, y - k1 / 3 + k2))
    k4 = h * np.array(f(t + h, y + k1 - k2 + k3))
    return y + (k1 + 3 * k2 + 3 * k3 + k4) / 8

def integrate_rk4(f, t_span, y0, h_int, h_print):
    t_start, t_end = t_span
    y = np.array(y0)
    t = t_start
    t_values = []
    y_values = []

    while t < t_end:
        t_values.append(t)
        y_values.append(y.copy())
        next_print_t = t + h_print
        if next_print_t > t_end:
            next_print_t = t_end

        steps = int((next_print_t - t) / h_int)
        for _ in range(steps):
            y = runge_kutta(f, t, y, h_int)
            t += h_int

        if t < next_print_t:
            h_last = next_print_t - t
            y = runge_kutta(f, t, y, h_last)
            t = next_print_t

    t_values.append(t)
    y_values.append(y.copy())

    return np.array(t_values), np.array(y_values)

# Интеграция для IVa (h_int=0.01)
t_iva, y_iva = integrate_rk4(system, t_span, y0, h_int=0.01, h_print=h_print)
x1_iva, x2_iva = y_iva[:, 0], y_iva[:, 1]

# Интеграция для IVб (h_int=0.005)
t_ivb, y_ivb = integrate_rk4(system, t_span, y0, h_int=0.005, h_print=h_print)
x1_ivb, x2_ivb = y_ivb[:, 0], y_ivb[:, 1]

# Сравнение результатов в таблице
print("Сравнение значений x1 и x2:")
print("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
    't', 'x1 (III)', 'x1 (IVa)', 'x1 (IVб)', 'x2 (III)', 'x2 (IVa)', 'x2 (IVб)'))

for i in range(len(t_eval)):
    t = t_eval[i]
    idx_iva = np.where(np.isclose(t_iva, t, atol=1e-6))[0][0]
    idx_ivb = np.where(np.isclose(t_ivb, t, atol=1e-6))[0][0]

    print("{:<8.2f} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f}".format(
        t,
        sol_rkf45.y[0][i], x1_iva[idx_iva], x1_ivb[idx_ivb],
        sol_rkf45.y[1][i], x2_iva[idx_iva], x2_ivb[idx_ivb]
    ))