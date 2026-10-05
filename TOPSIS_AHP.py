import numpy as np
import pandas as pd
from numpy.linalg import eig

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# ВХОДНЫЕ ДАННЫЕ
# ============================================================

# Альтернативы
alternatives = [
    "A1 Планировщик питания",
    "A2 Программирование",
    "A3 Трекер привычек",
    "A4 Расчет финансов",
]

# Критерии
criteria = [
    "C1 Востребованность",
    "C2 Простота",
    "C3 Интерес",
    "C4 Защищённость",
    "C5 Масштабируемость",
]

# Направление критериев: +1 = max (чем больше, тем лучше), -1 = min
directions = np.array([1, 1, 1, 1, 1])

# ------------------------------------------------------------
# Оценки важности критериев (3 эксперта × 5 критериев × 3 точки)
# Формат: [эксперт][критерий][p, e, o]
# ------------------------------------------------------------
weights_experts = np.array(
    [
        # Эксперт 1
        [
            [8, 9, 10],  # C1
            [8, 9, 10],  # C2
            [5, 6, 7],  # C3
            [7, 8, 9],  # C4
            [4, 5, 6],  # C5
        ],
        # Эксперт 2
        [
            [8, 9, 10],
            [7, 9, 10],
            [5, 7, 8],
            [7, 9, 10],
            [4, 6, 7],
        ],
        # Эксперт 3
        [
            [9, 10, 10],
            [8, 9, 10],
            [6, 7, 8],
            [8, 9, 10],
            [5, 6, 7],
        ],
    ],
    dtype=float,
)

# ------------------------------------------------------------
# Оценки альтернатив (3 эксперта × 4 альт × 5 критериев × 3 точки)
# ------------------------------------------------------------
ratings_experts = np.array(
    [
        # Эксперт 1
        [
            [[8, 9, 10], [8, 9, 10], [6, 7, 8], [8, 9, 10], [6, 7, 8]],  # A1
            [[8, 9, 10], [3, 4, 5], [8, 9, 10], [5, 6, 7], [7, 8, 9]],  # A2
            [[7, 8, 9], [8, 9, 10], [5, 6, 7], [8, 9, 10], [6, 7, 8]],  # A3
            [[6, 7, 8], [8, 9, 10], [5, 6, 7], [8, 9, 10], [5, 6, 7]],  # A4
        ],
        # Эксперт 2
        [
            [[7, 9, 10], [8, 9, 10], [6, 8, 9], [8, 9, 10], [6, 8, 9]],
            [[7, 9, 10], [3, 5, 6], [8, 9, 10], [5, 7, 8], [7, 9, 10]],
            [[6, 8, 9], [8, 9, 10], [6, 7, 8], [8, 9, 10], [6, 7, 8]],
            [[5, 7, 8], [8, 9, 10], [5, 6, 7], [8, 10, 10], [5, 7, 8]],
        ],
        # Эксперт 3
        [
            [[8, 10, 10], [8, 10, 10], [6, 7, 8], [8, 10, 10], [6, 7, 8]],
            [[8, 10, 10], [3, 4, 6], [8, 10, 10], [5, 6, 8], [7, 8, 10]],
            [[7, 8, 10], [8, 10, 10], [5, 7, 8], [8, 9, 10], [6, 8, 9]],
            [[6, 8, 9], [8, 10, 10], [5, 6, 8], [8, 9, 10], [5, 6, 8]],
        ],
    ],
    dtype=float,
)

# Агрегация: среднее по экспертам
weights_agg = weights_experts.mean(axis=0)  # (5, 3)
ratings_agg = ratings_experts.mean(axis=0)  # (4, 5, 3)

print("Агрегированные веса критериев (p, e, o):")
print(pd.DataFrame(weights_agg, index=criteria, columns=["p", "e", "o"]))
print()

print("Агрегированные оценки альтернатив:")
for i, alt in enumerate(alternatives):
    print(f"\n{alt}:")
    print(pd.DataFrame(ratings_agg[i], index=criteria, columns=["p", "e", "o"]))


# ============================================================
# НОРМИРОВКА ВЕСОВ
# ============================================================


def normalize_weights(weights):
    """Нормирует веса так, чтобы сумма по e была = 1."""
    w = weights.copy()
    sum_e = w[:, 1].sum()
    w = w / sum_e
    return w


weights_norm = normalize_weights(weights_agg)

print("Нормированные веса (p, e, o):")
print(pd.DataFrame(weights_norm, index=criteria, columns=["p", "e", "o"]))
print("Сумма по e:", weights_norm[:, 1].sum())

# ============================================================
# НЕЧЁТКИЙ TOPSIS
# ============================================================


def fuzzy_distance(a, b):
    """Расстояние между двумя треугольными нечёткими числами (vertex method)."""
    return np.sqrt(np.mean((a - b) ** 2))


def topsis(ratings, weights, directions):
    """
    ratings: (n_alt, n_crit, 3) — оценки (p, e, o)
    weights: (n_crit, 3) — веса (p, e, o)
    directions: (n_crit,) — +1 (max) или -1 (min)
    Возвращает: CC (n_alt,), d_plus, d_minus
    """
    n_alt, n_crit, _ = ratings.shape

    # --- Шаг 1. Нормализация ---
    normalized = np.zeros_like(ratings)
    for j in range(n_crit):
        if directions[j] == 1:  # max
            max_o = ratings[:, j, 2].max()
            normalized[:, j, :] = ratings[:, j, :] / max_o
        else:  # min
            min_p = ratings[:, j, 0].min()
            # для min: (min_p / o, min_p / e, min_p / p)
            normalized[:, j, 0] = min_p / ratings[:, j, 2]
            normalized[:, j, 1] = min_p / ratings[:, j, 1]
            normalized[:, j, 2] = min_p / ratings[:, j, 0]

    # --- Шаг 2. Взвешивание ---
    weighted = normalized * weights  # broadcasting (n_alt, n_crit, 3)

    # --- Шаг 3. Идеал и антиидеал ---
    V_plus = weighted.max(axis=0)  # (n_crit, 3)
    V_minus = weighted.min(axis=0)  # (n_crit, 3)

    # --- Шаг 4. Расстояния ---
    d_plus = np.zeros(n_alt)
    d_minus = np.zeros(n_alt)
    for i in range(n_alt):
        d_plus[i] = np.sqrt(
            sum(fuzzy_distance(weighted[i, j], V_plus[j]) ** 2 for j in range(n_crit))
        )
        d_minus[i] = np.sqrt(
            sum(fuzzy_distance(weighted[i, j], V_minus[j]) ** 2 for j in range(n_crit))
        )

    # --- Шаг 5. Коэффициент близости ---
    CC = d_minus / (d_plus + d_minus)

    return CC, d_plus, d_minus


# Расчёт
CC, d_plus, d_minus = topsis(ratings_agg, weights_norm, directions)

# Результаты
topsis_results = pd.DataFrame(
    {
        "Альтернатива": alternatives,
        "d+": d_plus,
        "d-": d_minus,
        "CC": CC,
        "Ранг": (-CC).argsort().argsort() + 1,
    }
).sort_values("CC", ascending=False)

print("=" * 60)
print("РЕЗУЛЬТАТЫ TOPSIS")
print("=" * 60)
print(topsis_results.to_string(index=False))

# ============================================================
# МАИ (AHP)
# ============================================================


def ahp_priority(matrix):
    """
    Вычисляет вектор приоритетов для матрицы парных сравнений.
    Возвращает: priority (n,), CR (consistency ratio)
    """
    n = matrix.shape[0]
    eigenvalues, eigenvectors = eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues.real[max_idx]
    priority = eigenvectors[:, max_idx].real
    priority = priority / priority.sum()

    # Индекс согласованности
    CI = (lambda_max - n) / (n - 1)
    RI_dict = {
        1: 0,
        2: 0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0

    return priority, CR


# ------------------------------------------------------------
# Матрица парных сравнений критериев (агрегированная)
# ------------------------------------------------------------
criteria_matrix = np.array(
    [
        [1, 1, 3, 1, 4],
        [1, 1, 3, 1, 4],
        [1 / 3, 1 / 3, 1, 1 / 3, 2],
        [1, 1, 3, 1, 3],
        [1 / 4, 1 / 4, 1 / 2, 1 / 3, 1],
    ]
)

criteria_priority, CR_crit = ahp_priority(criteria_matrix)
print("=" * 60)
print("МАИ: ВЕСА КРИТЕРИЕВ")
print("=" * 60)
for c, w in zip(criteria, criteria_priority):
    print(f"{c:30s}: {w:.4f}")
print(f"CR = {CR_crit:.4f} {'✅' if CR_crit < 0.1 else '❌'}")
print()

# ------------------------------------------------------------
# Матрицы парных сравнений альтернатив по каждому критерию
# ------------------------------------------------------------
alt_matrices = {
    "C1 Востребованность": np.array(
        [
            [1, 1, 2, 3],
            [1, 1, 2, 3],
            [1 / 2, 1 / 2, 1, 2],
            [1 / 3, 1 / 3, 1 / 2, 1],
        ]
    ),
    "C2 Простота": np.array(
        [
            [1, 7, 1, 1],
            [1 / 7, 1, 1 / 7, 1 / 7],
            [1, 7, 1, 1],
            [1, 7, 1, 1],
        ]
    ),
    "C3 Интерес": np.array(
        [
            [1, 1 / 3, 2, 3],
            [3, 1, 4, 5],
            [1 / 2, 1 / 4, 1, 2],
            [1 / 3, 1 / 5, 1 / 2, 1],
        ]
    ),
    "C4 Защищённость": np.array(
        [
            [1, 5, 1, 1],
            [1 / 5, 1, 1 / 5, 1 / 5],
            [1, 5, 1, 1],
            [1, 5, 1, 1],
        ]
    ),
    "C5 Масштабируемость": np.array(
        [
            [1, 1 / 2, 1, 2],
            [2, 1, 2, 3],
            [1, 1 / 2, 1, 2],
            [1 / 2, 1 / 3, 1 / 2, 1],
        ]
    ),
}

# Локальные приоритеты альтернатив по каждому критерию
alt_priorities = {}
print("=" * 60)
print("МАИ: ЛОКАЛЬНЫЕ ПРИОРИТЕТЫ АЛЬТЕРНАТИВ")
print("=" * 60)
for crit_name, matrix in alt_matrices.items():
    priority, CR = ahp_priority(matrix)
    alt_priorities[crit_name] = priority
    print(f"\n{crit_name}:")
    for alt, p in zip(alternatives, priority):
        print(f"  {alt:20s}: {p:.4f}")
    print(f"  CR = {CR:.4f} {'✅' if CR < 0.1 else '❌'}")

# ------------------------------------------------------------
# Синтез глобальных приоритетов
# ------------------------------------------------------------
global_priorities = np.zeros(len(alternatives))
for j, crit_name in enumerate(criteria):
    global_priorities += criteria_priority[j] * alt_priorities[crit_name]

# Результаты
ahp_results = pd.DataFrame(
    {
        "Альтернатива": alternatives,
        "Глобальный приоритет": global_priorities,
        "Ранг": (-global_priorities).argsort().argsort() + 1,
    }
).sort_values("Глобальный приоритет", ascending=False)

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ МАИ")
print("=" * 60)
print(ahp_results.to_string(index=False))

# ============================================================
# СРАВНЕНИЕ TOPSIS И МАИ
# ============================================================

comparison = pd.DataFrame(
    {
        "Альтернатива": alternatives,
        "TOPSIS CC": CC,
        "TOPSIS ранг": (-CC).argsort().argsort() + 1,
        "МАИ приоритет": global_priorities,
        "МАИ ранг": (-global_priorities).argsort().argsort() + 1,
    }
).sort_values("TOPSIS CC", ascending=False)

print("\n" + "=" * 60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)
print(comparison.to_string(index=False))

# Проверка совпадения лидера
topsis_winner = alternatives[np.argmax(CC)]
ahp_winner = alternatives[np.argmax(global_priorities)]

print(f"\nПобедитель TOPSIS: {topsis_winner}")
print(f"Победитель МАИ:    {ahp_winner}")

if topsis_winner == ahp_winner:
    print("Оба метода дают одинакового лидера")
else:
    print("Методы дают разных лидеров — нужен анализ.")
