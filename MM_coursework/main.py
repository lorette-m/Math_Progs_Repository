import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd


# Функция следа Якоби
def trace_eq(x2, x1):
    part1 = (1 + x2) * (2.97 * x1 ** 3 / ((1 + x1 ** 3) * (0.01 + x1 ** 3)) - 1)
    p4 = x1 * (1.5 + x2) / x2
    return part1 + (x1 - p4)


# Функция для вычисления определителя Якоби
def det_J(x1, x2, p1, p4):
    a11 = p1 * 3 * x1 ** 2 * 0.99 / (1 + x1 ** 3) ** 2 - (1 + x2)
    a12 = -x1
    a21 = 1.5 + x2
    a22 = x1 - p4
    return a11 * a22 - a12 * a21


# Функция для вычисления правой части первого уравнения
def dx1_dt(x1, x2, p1):
    return p1 * (0.01 + x1 ** 3) / (1 + x1 ** 3) - x1 * (1 + x2)


# Функция для вычисления правой части второго уравнения
def dx2_dt(x1, x2, p4):
    return x1 * (1.5 + x2) - p4 * x2


# Функция для вычисления следа Якоби
def trace_J(x1, x2, p1, p4):
    a11 = p1 * 3 * x1 ** 2 * 0.99 / (1 + x1 ** 3) ** 2 - (1 + x2)
    a22 = x1 - p4
    return a11 + a22

x1_values = np.arange(0.1, 1.41, 0.01)
p1_values = []
p4_values = []
p1_extra = []
p4_extra = []
table_data = []
last_x2 = 0.6  # Начальное приближение на основе расчёта

for x1 in x1_values:
    try:
        x2_sol, info, ier, mesg = fsolve(
            trace_eq, last_x2, args=(x1), full_output=True, xtol=1e-8, maxfev=1000
        )
        x2 = x2_sol[0]

        if ier == 1 and x2 > 0:
            p4 = x1 * (1.5 + x2) / x2
            if p4 > x1 and p4 > 0:
                p1 = x1 * (1 + x2) * (1 + x1 ** 3) / (0.01 + x1 ** 3)
                if p1 > 0:
                    det = det_J(x1, x2, p1, p4)
                    status = "Бифуркация" if det > 0 else "Лишняя"
                    table_data.append([x1, x2, p1, p4, det, status])
                    if det > 0:
                        p4_values.append(p4)
                        p1_values.append(p1)
                    else:
                        p4_extra.append(p4)
                        p1_extra.append(p1)
                    last_x2 = x2  # Обновляем приближение
    except:
        continue

df = pd.DataFrame(
    table_data,
    columns=["x1", "x2", "p1", "p4", "det_J", "Статус"]
)
df = df.round(4)  # Округляем для удобства чтения
df.to_csv("bifurcation_table.csv", index=False, encoding='utf-8')

plt.figure(figsize=(10, 6))
plt.plot(p4_values, p1_values, 'b--', alpha=0.5, label='Кривая бифуркации Хопфа')
plt.scatter(p4_values, p1_values, color='blue', marker='o', s=50, label='Точки бифуркации')
plt.scatter(p4_extra, p1_extra, color='red', marker='x', s=50, label='Лишние точки')
plt.xlabel('$p_4$')
plt.ylabel('$p_1$')
plt.title('Диаграмма бифуркации в плоскости $(p_4, p_1)$')
plt.grid(True)
plt.legend()
plt.savefig('hopf_bifurcation.png')
plt.close()

# Проверка для x1 = 0.2
x1_check = 0.2
print(f"\nПроверка для x1 = {x1_check:.2f}:")

# Находим x2 с помощью fsolve с начальным приближением 0.6
x2_sol_check, info, ier, mesg = fsolve(
    trace_eq, 0.6, args=(x1_check), full_output=True, xtol=1e-8, maxfev=1000
)
x2_check = x2_sol_check[0]

if ier == 1 and x2_check > 0:
    p4_check = x1_check * (1.5 + x2_check) / x2_check
    p1_check = x1_check * (1 + x2_check) * (1 + x1_check ** 3) / (0.01 + x1_check ** 3)

    # Вычисляем правые части уравнений
    dx1 = dx1_dt(x1_check, x2_check, p1_check)
    dx2 = dx2_dt(x1_check, x2_check, p4_check)

    # Вычисляем след и определитель Якоби
    trace = trace_J(x1_check, x2_check, p1_check, p4_check)
    det = det_J(x1_check, x2_check, p1_check, p4_check)

    print(f"x2 = {x2_check:.6f}")
    print(f"p1 = {p1_check:.6f}")
    print(f"p4 = {p4_check:.6f}")
    print(f"dx1/dt = {dx1:.6e} (должно быть близко к 0)")
    print(f"dx2/dt = {dx2:.6e} (должно быть близко к 0)")
    print(f"Trace(J) = {trace:.6e} (должно быть близко к 0)")
    print(f"Det(J) = {det:.6f} (должно быть > 0)")
else:
    print(f"Не удалось найти подходящее x2 для проверки: ier={ier}, x2={x2_check:.6f}, сообщение: {mesg}")