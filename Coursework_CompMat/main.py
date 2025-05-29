import numpy as np
from scipy import linalg
from scipy.integrate import quad, solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

print("Шаг 1: Вычисление R, R_2, E_2")
A = np.array([[46, 42, 24],
              [42, 49, 18],
              [24, 18, 16]])
b = np.array([2704, 2678, 1336])
lu, piv = linalg.lu_factor(A)
x = linalg.lu_solve((lu, piv), b)
R, R2, E2 = x[0], x[1], x[2]
R1, R3 = R, R
print(f"R = R1 = R3 = {R:.6f} Ом")
print(f"R2 = {R2:.6f} Ом")
print(f"E2 = {E2:.6f} В")

# Шаг 2: Вычисление L
print("\nШаг 2: Вычисление L")
def integrand(x):
    return x * np.log10(x)

integral, error = quad(integrand, 1, 2)
coeff_L = 0.1447496
L = coeff_L * integral
L1, L3 = L, L
print(f"Интеграл: {integral:.6f}, Погрешность интеграла: {error:.2e}")
print(f"L = L1 = L3 = {L:.6f} Гн")

# Шаг 3: Вычисление E_1
print("\nШаг 3: Вычисление E_1")
def f(x):
    return np.exp(-x) - (x - 1)**2
solution = root_scalar(f, bracket=[1, 2], method='brentq')
x_star = solution.root
coeff_E1 = 2.706964
E1 = coeff_E1 * x_star
print(f"x* = {x_star:.6f}")
print(f"E1 = {E1:.6f} В")
print(f"Проверка: f(x*) = {f(x_star):.2e}")

# Шаг 4: Параметры цепи и начальные условия
print("\nШаг 4: Параметры цепи и начальные условия")
C = 1e-6  # Фарад (из условия)
print(f"C = {C:.2e} Ф")
print(f"Используемые параметры:")
print(f"R1 = {R1:.6f} Ом, R2 = {R2:.6f} Ом, R3 = {R3:.6f} Ом")
print(f"L1 = {L1:.6f} Гн, L3 = {L3:.6f} Гн")
print(f"E1 = {E1:.6f} В, E2 = {E2:.6f} В")

# Начальные условия для основного расчёта
I1 = E1 / R1  # Ток i1 при открытом ключе
I3 = 0  # Ток i3 при открытом ключе
Uc_0 = -1 * E2  # Напряжение на конденсаторе при открытом ключе
y0 = [I1, I3, Uc_0]  # [i1, i3, Uc]
print(f"Начальные условия (основной): i1(0) = {I1:.6f} А, i3(0) = {I3:.6f} А, Uc(0) = {Uc_0:.6f} В")

# Начальные условия для второго расчёта (Uc увеличено на 1%)
Uc_0_new = Uc_0 * 1.01
y0_new = [I1, I3, Uc_0_new]
print(f"Начальные условия (второй случай): i1(0) = {I1:.6f} А, i3(0) = {I3:.6f} А, Uc(0) = {Uc_0_new:.6f} В")

# Шаг 5: Решение системы дифференциальных уравнений
print("\nШаг 5: Решение системы дифференциальных уравнений")
t_span = (0, 0.015)  # 15 мс
t_eval = np.linspace(0, 0.015, 1000)

def system(t, y):
    i1, i3, Uc = y
    di1_dt = (1 / L1) * (E1 - E2 - Uc + i3 * R2 - i1 * (R1 + R2))
    di3_dt = (1 / L3) * (E2 + Uc + i1 * R2 - i3 * (R2 + R3))
    dUc_dt = (1 / C) * (i1 - i3)
    return [di1_dt, di3_dt, dUc_dt]

# Основное решение
sol = solve_ivp(system, t_span, y0, method='RK45', t_eval=t_eval)
t = sol.t
i1, i3, Uc = sol.y[0], sol.y[1], sol.y[2]

# Второе решение с изменённым Uc(0)
sol_new = solve_ivp(system, t_span, y0_new, method='RK45', t_eval=t_eval)
i1_new, i3_new, Uc_new = sol_new.y[0], sol_new.y[1], sol_new.y[2]

# Шаг 6: Построение графиков
print("\nШаг 6: Построение графиков")
plt.figure(figsize=(12, 10))

# График i1(t)
plt.subplot(3, 1, 1)
plt.plot(t, i1, label='i1(t) (основной)', color='blue')
plt.plot(t, i1_new, label='i1(t) (Uc +1%)', color='red', linestyle='--')
plt.xlabel('Время (с)')
plt.ylabel('Ток (А)')
plt.legend()
plt.grid(True)

# График i3(t)
plt.subplot(3, 1, 2)
plt.plot(t, i3, label='i3(t) (основной)', color='blue')
plt.plot(t, i3_new, label='i3(t) (Uc +1%)', color='red', linestyle='--')
plt.xlabel('Время (с)')
plt.ylabel('Ток (А)')
plt.legend()
plt.grid(True)

# График Uc(t)
plt.subplot(3, 1, 3)
plt.plot(t, Uc, label='Uc(t) (основной)', color='blue')
plt.plot(t, Uc_new, label='Uc(t) (Uc +1%)', color='red', linestyle='--')
plt.xlabel('Время (с)')
plt.ylabel('Напряжение (В)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Шаг 7: Оценка погрешности (для основного решения)
print("\nШаг 7: Оценка погрешности (основной расчёт)")
sol_precise = solve_ivp(system, t_span, y0, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-8)
error_i1 = np.max(np.abs(sol.y[0] - sol_precise.y[0]))
error_i3 = np.max(np.abs(sol.y[1] - sol_precise.y[1]))
error_Uc = np.max(np.abs(sol.y[2] - sol_precise.y[2]))
print(f"Максимальная погрешность i1: {error_i1:.2e} А")
print(f"Максимальная погрешность i3: {error_i3:.2e} А")
print(f"Максимальная погрешность Uc: {error_Uc:.2e} В")

# Шаг 8: Построение таблицы значений
print("\nШаг 8: Таблица значений для ключевых точек времени")
key_times = np.array([0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.015])
indices = np.searchsorted(t, key_times, side='left')
print(f"{'t (с)':<10} {'i1 (А)':<15} {'i1_new (А)':<15} {'i3 (А)':<15} {'i3_new (А)':<15} {'Uc (В)':<15} {'Uc_new (В)':<15}")
for idx in indices:
    print(f"{t[idx]:<10.3f} {i1[idx]:<15.6f} {i1_new[idx]:<15.6f} {i3[idx]:<15.6f} {i3_new[idx]:<15.6f} {Uc[idx]:<15.6f} {Uc_new[idx]:<15.6f}")
