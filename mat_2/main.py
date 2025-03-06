import numpy as np
from scipy.linalg import lu_factor, lu_solve

p_values = [1.0, 0.1, 0.01, 0.0001, 0.000001]

A_elements = np.array([
    [31, -7, -7, -4, -8, -4, -1, 0],
    [-7, 27, -4, -4, 0, -1, -4, -7],
    [-7, -4, 31, -8, -7, 0, -3, -2],
    [-4, -4, -8, 39, -4, -7, -7, -5],
    [-8, 0, -7, -4, 29, -7, -2, -1],
    [-4, -1, 0, -7, -7, 25, -1, -5],
    [-1, -4, -3, -7, -2, -1, 20, -2],
    [0, -7, -2, -5, -1, -5, -2, 22]
], dtype=float)

b_elements = np.array([-29, -28, -104, 198, 29, 59, -71, -54], dtype=float)

results = []

for idx, p in enumerate(p_values):
    A = A_elements.copy()
    A[0, 0] += p

    b = b_elements.copy()
    b[0] += 3 * p

    lu, piv = lu_factor(A)
    lu_solution = lu_solve((lu, piv), b)

    # обратная матрица
    n = A.shape[0]
    I = np.eye(n)
    A_inv = np.zeros_like(A)
    for i in range(n):
        A_inv[:, i] = lu_solve((lu, piv), I[:, i])

    # x1 = A^-1 * b
    x1 = A_inv @ b

    cond_A = np.linalg.cond(A, p=np.inf)

    # δ = ||x1 - x2|| / ||x1||
    delta = np.linalg.norm(x1 - lu_solution) / np.linalg.norm(x1)

    diff = x1 - lu_solution

    results.append((p, cond_A, delta, x1, lu_solution, diff))

# Консольный вывод
for idx, (p, cond, delta, x1, lu_solution, diff) in enumerate(results):
    print(f"\n=== Результаты для p = {p:.6f} ===")
    print(f"Число обусловленности (cond): {cond:.6e}")
    print(f"Относительная погрешность (δ): {delta:.6e}")

    # Форматированный вывод векторов
    print("\nВектор x1 (A⁻¹b):")
    print(np.array2string(x1, precision=6, suppress_small=True, floatmode='fixed'))

    print("\nВектор x2 (LU-решение):")
    print(np.array2string(lu_solution, precision=6, suppress_small=True, floatmode='fixed'))

    print("\n" + "=" * 50)

# График зависимости δ от cond
import matplotlib.pyplot as plt

conds = [res[1] for res in results]
deltas = [res[2] for res in results]

plt.figure(figsize=(10, 6))
plt.plot(conds, deltas, 'bo-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Число обусловленности (cond)')
plt.ylabel('Относительная погрешность (δ)')
plt.title('Зависимость δ от cond')
plt.grid(True)
plt.show()