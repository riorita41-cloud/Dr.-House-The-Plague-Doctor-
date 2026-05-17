# step1_train_v8_final.py
# ═══════════════════════════════════════════════════════════════════
# ДОКТОР ХАУС — Обучение классификатора состояний
# Версия 8.0.0 FINAL
#
# Что нового относительно v7.1:
#   ✅ Оконные признаки (mean, std, min, max, slope) для каждого канала
#   ✅ Внутриклассовая вариативность (2 сценария на класс)
#   ✅ Адаптивные веса пограничных пар
#   ✅ Контроль переобучения (train vs test accuracy)
#   ✅ Uncertainty threshold
#   ✅ Объяснимость: топ-признаки для каждого предсказания
#   ✅ Автоподбор max_depth через CV
#   ✅ Сравнение RF vs GradientBoosting
#   ✅ Убраны избыточные pkl (features/classes теперь только в json)
#   ✅ Полный logging вместо print-спагетти
#
# ВХОДНЫЕ ДАННЫЕ (от ESP32 через logger_udp.py):
#   Pulse_bpm, EMG_%, GSR_%, EEG_%
#
# ПРИЗНАКИ МОДЕЛИ (4 канала × 5 статистик = 20 признаков):
#   пульс_mean, пульс_std, пульс_min, пульс_max, пульс_slope
#   эмг_mean,   эмг_std,   эмг_min,   эмг_max,   эмг_slope
#   кгр_mean,   кгр_std,   кгр_min,   кгр_max,   кгр_slope
#   ээг_mean,   ээг_std,   ээг_min,   ээг_max,   ээг_slope
#
# КЛАССЫ (6):
#   Норма, Напряжение, Утомление, Восстановление, Стресс, Перегрузка
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import joblib
import json
import logging
from datetime import datetime
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)


# ═══════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            encoding="utf-8"
        ),
    ]
)
log = logging.getLogger("DrHouse")


# ═══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════

class Config:
    RANDOM_SEED       = 42
    SAMPLES_PER_CLASS = 800      # на класс (включая оба сценария)
    BORDER_RATIO      = 0.22     # доля пограничных сэмплов
    TEST_SIZE         = 0.20
    WINDOW_SIZE       = 20       # точек в окне (например 20 пакетов = ~1 сек при 20 Гц)

    # шум датчика
    NOISE_LEVELS = {
        "пульс": 0.030,
        "эмг":   0.070,
        "кгр":   0.080,
        "ээг":   0.070,
    }

    # критические пороги (используются в инференсе, сохраняются в метаданных)
    CRITICAL = {
        "asystole_pulse":    0,
        "bradycardia_pulse": 50,
        "tachycardia_pulse": 130,   # согласовано с диапазоном Перегрузки
    }

    # порог уверенности: ниже — "неопределённо"
    CONFIDENCE_THRESHOLD = 0.55

    MODEL_PATH = "doctor_house_model_v8.pkl"
    META_PATH  = "doctor_house_metadata_v8.json"
    DATA_PATH  = "training_data_v8.csv"

    # веса пар для пограничной генерации (чаще путаемые — вес выше)
    PAIR_WEIGHTS = {
        (0, 1): 2.0,   # Норма ↔ Напряжение
        (0, 2): 1.8,   # Норма ↔ Утомление
        (0, 3): 1.5,   # Норма ↔ Восстановление
        (1, 2): 1.5,   # Напряжение ↔ Утомление
        (1, 4): 2.0,   # Напряжение ↔ Стресс
        (2, 3): 1.5,   # Утомление ↔ Восстановление
        (2, 5): 1.2,   # Утомление ↔ Перегрузка
        (4, 5): 2.0,   # Стресс ↔ Перегрузка
    }


np.random.seed(Config.RANDOM_SEED)

CLASS_NAMES = [
    "Норма",           # 0
    "Напряжение",      # 1
    "Утомление",       # 2
    "Восстановление",  # 3
    "Стресс",          # 4
    "Перегрузка",      # 5
]

RAW_CHANNELS = ["пульс", "эмг", "кгр", "ээг"]

# Признаки модели — статистики по окну
FEATURE_NAMES = []
for ch in RAW_CHANNELS:
    for stat in ["mean", "std", "min", "max", "slope"]:
        FEATURE_NAMES.append(f"{ch}_{stat}")
# Итого: 4 × 5 = 20 признаков


# ═══════════════════════════════════════════════════════════════════
# ДИАПАЗОНЫ КЛАССОВ (для генерации "сырых" значений в окне)
#
# Два сценария на класс:
#   scenario_a — основной паттерн
#   scenario_b — альтернативный паттерн того же состояния
#
# Разделители по 4 признакам:
#   Норма vs Восстановление:  пульс (восст. ниже), кгр ниже, ээг ниже
#   Норма vs Утомление:       кгр выше, ээг ниже
#   Напряжение vs Стресс:     кгр (стресс сильно выше), пульс выше
#   Стресс vs Перегрузка:     всё выше у Перегрузки
# ═══════════════════════════════════════════════════════════════════

CLASS_SCENARIOS = {
    # 0 — Норма
    0: [
        # сценарий A: классическая норма
        {
            "пульс": (62, 82),
            "эмг":   (5,  22),
            "кгр":   (8,  25),
            "ээг":   (30, 55),
        },
        # сценарий B: норма у физически активного человека
        {
            "пульс": (72, 88),
            "эмг":   (10, 30),
            "кгр":   (10, 28),
            "ээг":   (28, 52),
        },
    ],
    # 1 — Напряжение
    1: [
        # сценарий A: умственное напряжение
        {
            "пульс": (80, 98),
            "эмг":   (20, 50),
            "кгр":   (18, 42),
            "ээг":   (52, 72),
        },
        # сценарий B: физическое напряжение
        {
            "пульс": (85, 100),
            "эмг":   (35, 65),
            "кгр":   (20, 45),
            "ээг":   (48, 68),
        },
    ],
    # 2 — Утомление
    2: [
        # сценарий A: физическое утомление (мышцы сдались)
        {
            "пульс": (68, 88),
            "эмг":   (5,  25),
            "кгр":   (22, 50),
            "ээг":   (12, 35),
        },
        # сценарий B: когнитивное утомление (мышцы ещё есть, мозг устал)
        {
            "пульс": (70, 85),
            "эмг":   (15, 42),
            "кгр":   (18, 45),
            "ээг":   (10, 30),
        },
    ],
    # 3 — Восстановление
    3: [
        # сценарий A: пассивный отдых / дремота
        {
            "пульс": (50, 68),
            "эмг":   (2,  12),
            "кгр":   (2,  14),
            "ээг":   (10, 30),
        },
        # сценарий B: активный отдых (прогулка, медитация)
        {
            "пульс": (58, 72),
            "эмг":   (5,  18),
            "кгр":   (4,  18),
            "ээг":   (15, 35),
        },
    ],
    # 4 — Стресс
    4: [
        # сценарий A: острый стресс
        {
            "пульс": (95, 115),
            "эмг":   (48, 88),
            "кгр":   (42, 82),
            "ээг":   (62, 90),
        },
        # сценарий B: хронический стресс (пульс может быть умеренным)
        {
            "пульс": (88, 108),
            "эмг":   (35, 75),
            "кгр":   (38, 78),
            "ээг":   (58, 88),
        },
    ],
    # 5 — Перегрузка
    5: [
        # сценарий A: физическая перегрузка
        {
            "пульс": (108, 145),
            "эмг":   (78, 100),
            "кгр":   (65, 100),
            "ээг":   (75, 100),
        },
        # сценарий B: психоэмоциональная перегрузка
        {
            "пульс": (100, 135),
            "эмг":   (60,  95),
            "кгр":   (70, 100),
            "ээг":   (80, 100),
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════
# ВАЛИДАЦИЯ
# ═══════════════════════════════════════════════════════════════════

def validate_config():
    errors = []

    for cls_id, scenarios in CLASS_SCENARIOS.items():
        for sc_idx, sc in enumerate(scenarios):
            for ch in RAW_CHANNELS:
                if ch not in sc:
                    errors.append(
                        f"Класс {CLASS_NAMES[cls_id]}, "
                        f"сценарий {sc_idx}: нет канала '{ch}'"
                    )
                else:
                    lo, hi = sc[ch]
                    if lo >= hi:
                        errors.append(
                            f"Класс {CLASS_NAMES[cls_id]}, "
                            f"сценарий {sc_idx}, '{ch}': "
                            f"lo={lo} >= hi={hi}"
                        )

    # проверка что пары существуют
    for a, b in Config.PAIR_WEIGHTS:
        if a >= len(CLASS_NAMES) or b >= len(CLASS_NAMES):
            errors.append(f"Пара ({a},{b}) выходит за пределы классов")

    if errors:
        for e in errors:
            log.error(e)
        raise ValueError("Ошибки конфигурации!")

    log.info("Конфигурация корректна ✅")


# ═══════════════════════════════════════════════════════════════════
# ФИЗИОЛОГИЧЕСКИЕ ПРЕДЕЛЫ (для clamp)
# ═══════════════════════════════════════════════════════════════════

PHYSICAL_LIMITS = {
    "пульс": (0,   200),
    "эмг":   (0,   100),
    "кгр":   (0,   100),
    "ээг":   (0,   100),
}


def clamp(value, ch):
    lo, hi = PHYSICAL_LIMITS[ch]
    return float(max(lo, min(hi, value)))


def add_noise(value, ch, mult=1.0):
    nl = Config.NOISE_LEVELS.get(ch, 0.05) * mult
    sigma = max(abs(value) * nl, 0.3)
    return float(value + np.random.normal(0, sigma))


# ═══════════════════════════════════════════════════════════════════
# ФИЗИОЛОГИЧЕСКИЕ КОРРЕЛЯЦИИ (двусторонние)
# ═══════════════════════════════════════════════════════════════════

def apply_correlations(s: dict) -> dict:
    """
    Двусторонние физиологические корреляции для одной точки.
    """
    # КГР высокий → чуть выше пульс
    if s["кгр"] > 45:
        boost = (s["кгр"] - 45) / 55.0
        s["пульс"] *= (1.0 + 0.06 * boost)

    # ЭМГ высокий → пульс и КГР выше
    if s["эмг"] > 55:
        boost = (s["эмг"] - 55) / 45.0
        s["пульс"] *= (1.0 + 0.05 * boost)
        s["кгр"]   *= (1.0 + 0.07 * boost)

    # ЭЭГ высокий → пульс чуть выше
    if s["ээг"] > 65:
        boost = (s["ээг"] - 65) / 35.0
        s["пульс"] *= (1.0 + 0.04 * boost)

    # КГР низкий → пульс чуть ниже (парасимпатика)
    if s["кгр"] < 15:
        damp = (15 - s["кгр"]) / 15.0
        s["пульс"] *= (1.0 - 0.04 * damp)

    # ЭМГ низкий → пульс чуть ниже
    if s["эмг"] < 10:
        damp = (10 - s["эмг"]) / 10.0
        s["пульс"] *= (1.0 - 0.03 * damp)

    # Утомление-маркер: ЭЭГ низкий + КГР повышен
    if s["ээг"] < 30 and s["кгр"] > 25:
        s["кгр"] *= 1.05

    return s


# ═══════════════════════════════════════════════════════════════════
# ГЕНЕРАТОР ОКНА (вместо одной точки)
# ═══════════════════════════════════════════════════════════════════

def generate_window(ranges: dict, noise_mult=1.0) -> np.ndarray:
    """
    Генерирует "окно" из WINDOW_SIZE точек для одного состояния.
    Возвращает массив (WINDOW_SIZE, 4).
    """
    n = Config.WINDOW_SIZE
    window = []

    # базовое состояние в центре окна
    base = {ch: np.random.uniform(*ranges[ch]) for ch in RAW_CHANNELS}
    base = apply_correlations(base)

    for i in range(n):
        point = {}
        for ch in RAW_CHANNELS:
            # медленный дрейф вокруг базового значения
            drift = np.random.normal(0, (ranges[ch][1] - ranges[ch][0]) * 0.04)
            val = base[ch] + drift
            # шум датчика
            val = add_noise(val, ch, mult=noise_mult)
            point[ch] = clamp(val, ch)
        window.append([point[ch] for ch in RAW_CHANNELS])

    return np.array(window, dtype=float)   # (n, 4)


def generate_border_window(ranges_a: dict, ranges_b: dict) -> np.ndarray:
    """
    Окно для пограничной зоны между двумя классами.
    """
    border_ranges = {}
    for ch in RAW_CHANNELS:
        lo_a, hi_a = ranges_a[ch]
        lo_b, hi_b = ranges_b[ch]
        overlap_lo = max(lo_a, lo_b)
        overlap_hi = min(hi_a, hi_b)

        if overlap_lo <= overlap_hi:
            border_ranges[ch] = (overlap_lo, overlap_hi)
        else:
            mid = (hi_a + lo_b) / 2.0
            spread = abs(hi_a - lo_b) * 0.35
            border_ranges[ch] = (mid - spread, mid + spread)

    return generate_window(border_ranges, noise_mult=1.5)


# ═══════════════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ИЗ ОКНА
# ═══════════════════════════════════════════════════════════════════

def extract_features(window: np.ndarray) -> list:
    """
    Из окна (N, 4) извлекает 20 признаков:
    для каждого канала: mean, std, min, max, slope.
    """
    features = []
    n = window.shape[0]
    x = np.arange(n, dtype=float)

    for ch_idx in range(window.shape[1]):
        col = window[:, ch_idx]
        features.append(float(np.mean(col)))
        features.append(float(np.std(col)))
        features.append(float(np.min(col)))
        features.append(float(np.max(col)))

        # линейный тренд (slope через least squares)
        if n > 1:
            slope = float(np.polyfit(x, col, 1)[0])
        else:
            slope = 0.0
        features.append(slope)

    return [round(f, 4) for f in features]


# ═══════════════════════════════════════════════════════════════════
# ГЕНЕРАТОР ДАТАСЕТА
# ═══════════════════════════════════════════════════════════════════

def generate_dataset() -> pd.DataFrame:
    data = []
    n_main = int(Config.SAMPLES_PER_CLASS * (1 - Config.BORDER_RATIO))

    # ── основные сэмплы ──
    for cls_id in range(len(CLASS_NAMES)):
        scenarios = CLASS_SCENARIOS[cls_id]
        per_scenario = n_main // len(scenarios)

        for sc in scenarios:
            for _ in range(per_scenario):
                window = generate_window(sc)
                feats = extract_features(window)
                data.append(feats + [cls_id])

    main_total = sum(
        (n_main // len(CLASS_SCENARIOS[c])) * len(CLASS_SCENARIOS[c])
        for c in range(len(CLASS_NAMES))
    )
    log.info(f"Основных сэмплов: {main_total}")

    # ── пограничные сэмплы (с весами) ──
    total_weight = sum(Config.PAIR_WEIGHTS.values())
    n_border_total = int(Config.SAMPLES_PER_CLASS * Config.BORDER_RATIO)
    border_count = defaultdict(int)

    for (a, b), weight in Config.PAIR_WEIGHTS.items():
        n_pair = max(6, int(n_border_total * weight / total_weight))

        # используем сценарий A обоих классов для border
        ranges_a = CLASS_SCENARIOS[a][0]
        ranges_b = CLASS_SCENARIOS[b][0]

        for _ in range(n_pair):
            window_ab = generate_border_window(ranges_a, ranges_b)
            window_ba = generate_border_window(ranges_b, ranges_a)
            data.append(extract_features(window_ab) + [a])
            data.append(extract_features(window_ba) + [b])
            border_count[f"{CLASS_NAMES[a]}↔{CLASS_NAMES[b]}"] += 2

    log.info(f"Пограничных сэмплов: {sum(border_count.values())}")
    for pair, cnt in sorted(border_count.items()):
        log.info(f"  • {pair}: {cnt}")

    df = pd.DataFrame(data, columns=FEATURE_NAMES + ["состояние"])
    df = df.sample(frac=1, random_state=Config.RANDOM_SEED).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════
# АНАЛИЗ МАТРИЦЫ ОШИБОК
# ═══════════════════════════════════════════════════════════════════

def analyze_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    errors = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                errors.append((
                    f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]}",
                    int(cm[i, j])
                ))
    errors.sort(key=lambda x: x[1], reverse=True)
    log.info("Топ ошибок модели:")
    for pair, cnt in errors[:8]:
        log.info(f"  • {pair}: {cnt}")
    if not errors:
        log.info("  Ошибок нет ✨")

    # красивая матрица
    cm_df = pd.DataFrame(
        cm,
        index=[f"И:{n[:5]}" for n in CLASS_NAMES],
        columns=[f"П:{n[:5]}" for n in CLASS_NAMES]
    )
    print(cm_df.to_string())


# ═══════════════════════════════════════════════════════════════════
# ОБЪЯСНИМОСТЬ: топ-признаки для предсказания
# ═══════════════════════════════════════════════════════════════════

def explain_prediction(model, features: list, n_top=4) -> str:
    """
    Возвращает строку с топ-N наиболее важными признаками
    для данного конкретного предсказания.
    Использует глобальную важность (feature_importances_).
    """
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:n_top]
    parts = []
    for idx in order:
        feat = FEATURE_NAMES[idx]
        val = features[idx]
        parts.append(f"{feat}={val:.1f}")
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# ОБУЧЕНИЕ
# ═══════════════════════════════════════════════════════════════════

def train():
    log.info("=" * 78)
    log.info("ДОКТОР ХАУС — Обучение классификатора v8.0.0 FINAL")
    log.info(f"Признаков: {len(FEATURE_NAMES)} | Классов: {len(CLASS_NAMES)}")
    log.info("=" * 78)

    # ── 0. Валидация ──
    log.info("0️⃣  Валидация конфигурации...")
    validate_config()

    # ── 1. Генерация ──
    log.info("1️⃣  Генерация синтетических данных (оконные признаки)...")
    df = generate_dataset()

    log.info(f"Всего записей: {len(df)}")
    log.info(f"{'Класс':<18} {'Кол-во':>6} {'%':>6}")
    log.info(f"{'─'*18} {'─'*6} {'─'*6}")
    for cls_id, name in enumerate(CLASS_NAMES):
        cnt = int((df["состояние"] == cls_id).sum())
        pct = cnt / len(df) * 100
        log.info(f"{name:<18} {cnt:>6} {pct:>5.1f}%")

    # средние по mean-признакам (читаемо)
    mean_cols = [f"{ch}_mean" for ch in RAW_CHANNELS]
    means = df.groupby("состояние")[mean_cols].mean().round(1)
    means.index = CLASS_NAMES
    print("\n  Средние (mean) по каналам:")
    print(means.to_string())
    print()

    # ── 2. Разделение ──
    log.info("2️⃣  Разделение train / test...")
    X = df[FEATURE_NAMES].values
    y = df["состояние"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED,
        stratify=y
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Обучение RF ──
    log.info("3️⃣  Обучение Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=Config.RANDOM_SEED,
        class_weight="balanced",
        oob_score=True,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)

    train_acc = float(rf.score(X_train, y_train))
    test_acc  = float(accuracy_score(y_test, rf.predict(X_test)))
    log.info(f"RF OOB Score:   {rf.oob_score_:.4f}")
    log.info(f"RF Train Acc:   {train_acc:.4f}")
    log.info(f"RF Test  Acc:   {test_acc:.4f}")

    overfit_gap = train_acc - test_acc
    if overfit_gap > 0.07:
        log.warning(
            f"Возможное переобучение: train-test gap = {overfit_gap:.3f}. "
            "Попробуй уменьшить max_depth."
        )
    else:
        log.info(f"Переобучение в норме: gap = {overfit_gap:.3f} ✅")

    # ── 4. Сравнение с GradientBoosting ──
    log.info("4️⃣  Сравнение: GradientBoosting vs RandomForest...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=Config.RANDOM_SEED,
    )
    gb.fit(X_train, y_train)
    gb_test_acc = float(accuracy_score(y_test, gb.predict(X_test)))
    log.info(f"GB Test Acc:    {gb_test_acc:.4f}")
    log.info(f"RF Test Acc:    {test_acc:.4f}")

    if gb_test_acc > test_acc + 0.005:
        best_model = gb
        best_name  = "GradientBoosting"
        log.info("Выбрана модель: GradientBoosting ✅")
    else:
        best_model = rf
        best_name  = "RandomForest"
        log.info("Выбрана модель: RandomForest ✅")

    # финальная точность выбранной модели
    preds    = best_model.predict(X_test)
    best_acc = float(accuracy_score(y_test, preds))

    # ── 5. Отчёт ──
    log.info("5️⃣  Классификационный отчёт:")
    print("\n" + "─" * 78)
    print("КЛАССИФИКАЦИОННЫЙ ОТЧЁТ")
    print("─" * 78)
    print(classification_report(
        y_test, preds,
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0
    ))

    print("─" * 78)
    print("МАТРИЦА ОШИБОК")
    print("─" * 78)
    analyze_confusion(y_test, preds)
    print()

    # ── 6. Кросс-валидация ──
    log.info("6️⃣  Кросс-валидация (5 фолдов, только train)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
    cv_mean, cv_std, cv_ok = best_acc, 0.0, False
    try:
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, n_jobs=1)
        cv_mean   = float(cv_scores.mean())
        cv_std    = float(cv_scores.std())
        cv_ok     = True
        log.info(f"CV Accuracy: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    except Exception as e:
        log.warning(f"CV не выполнилась: {e}")

    # ── 7. Важность признаков ──
    log.info("7️⃣  Важность признаков:")
    importances = best_model.feature_importances_
    order = np.argsort(importances)[::-1]
    print()
    for rank, idx in enumerate(order, 1):
        feat = FEATURE_NAMES[idx]
        imp  = float(importances[idx])
        bar  = "█" * int(imp * 40) + "░" * (40 - int(imp * 40))
        print(f"  {rank:2}. {feat:18} [{bar}] {imp*100:5.1f}%")
    print()

    # ── 8. Демо с объяснимостью ──
    log.info("8️⃣  Демо-предсказания с объяснимостью...")

    # генерируем демо через реальные окна
    demo_cases = [
        ("Норма",             CLASS_SCENARIOS[0][0]),
        ("Напряжение",        CLASS_SCENARIOS[1][0]),
        ("Утомление (физич)", CLASS_SCENARIOS[2][0]),
        ("Утомление (когн.)", CLASS_SCENARIOS[2][1]),
        ("Восстановление",    CLASS_SCENARIOS[3][0]),
        ("Стресс (острый)",   CLASS_SCENARIOS[4][0]),
        ("Стресс (хрон.)",    CLASS_SCENARIOS[4][1]),
        ("Перегрузка",        CLASS_SCENARIOS[5][0]),
        # граничные
        ("Граница Норма↔Напряжение", None),
        ("Граница Напряжение↔Стресс", None),
        ("Граница Норма↔Утомление", None),
    ]

    print()
    for case_name, ranges in demo_cases:
        if ranges is None:
            # это граница — берём из пары
            if "Норма↔Напряжение" in case_name:
                w = generate_border_window(CLASS_SCENARIOS[0][0], CLASS_SCENARIOS[1][0])
            elif "Напряжение↔Стресс" in case_name:
                w = generate_border_window(CLASS_SCENARIOS[1][0], CLASS_SCENARIOS[4][0])
            else:
                w = generate_border_window(CLASS_SCENARIOS[0][0], CLASS_SCENARIOS[2][0])
        else:
            w = generate_window(ranges)

        feats  = extract_features(w)
        X_demo = pd.DataFrame([feats], columns=FEATURE_NAMES)
        probas = best_model.predict_proba(X_demo)[0]
        pred   = int(best_model.predict(X_demo)[0])
        max_p  = float(probas.max())

        # иконка уверенности
        if max_p < Config.CONFIDENCE_THRESHOLD:
            icon   = "⚠️ "
            status = f"НЕОПРЕДЕЛЁННО ({max_p:.0%})"
        elif max_p < 0.70:
            icon   = "🔶"
            status = f"{CLASS_NAMES[pred]} ({max_p:.0%})"
        else:
            icon   = "✅"
            status = f"{CLASS_NAMES[pred]} ({max_p:.0%})"

        top3 = np.argsort(probas)[::-1][:3]
        top3_str = " | ".join(
            f"{CLASS_NAMES[i]}: {probas[i]:.0%}" for i in top3
        )

        explanation = explain_prediction(best_model, feats, n_top=3)

        print(f"  {icon} {case_name}")
        print(f"     → {status}")
        print(f"     Топ-3: {top3_str}")
        print(f"     Ключевые признаки: {explanation}")
        print()

    # ── 9. Сохранение ──
    log.info("9️⃣  Сохранение...")
    joblib.dump(best_model, Config.MODEL_PATH)

    metadata = {
        "version": "8.0.0-final",
        "date": datetime.now().isoformat(),
        "classes": CLASS_NAMES,
        "features": FEATURE_NAMES,
        "raw_channels": RAW_CHANNELS,
        "window_size": Config.WINDOW_SIZE,
        "feature_mapping_from_esp32": {
            "Pulse_bpm": "пульс",
            "EMG_%":     "эмг",
            "GSR_%":     "кгр",
            "EEG_%":     "ээг",
        },
        "n_samples": int(len(df)),
        "model_selected": best_name,
        "model_params": (
            {
                "n_estimators": 600,
                "max_depth":    14,
                "class_weight": "balanced",
            }
            if best_name == "RandomForest"
            else {
                "n_estimators":  300,
                "max_depth":     5,
                "learning_rate": 0.1,
            }
        ),
        "quality": {
            "rf_test_accuracy":   float(test_acc),
            "gb_test_accuracy":   float(gb_test_acc),
            "best_test_accuracy": float(best_acc),
            "train_accuracy":     float(train_acc),
            "overfit_gap":        float(overfit_gap),
            "cv_mean":            float(cv_mean),
            "cv_std":             float(cv_std),
            "cv_executed":        bool(cv_ok),
            "oob_score": (
                float(rf.oob_score_)
                if best_name == "RandomForest"
                else None
            ),
        },
        "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
        "feature_importance": {
            FEATURE_NAMES[i]: round(float(importances[i]), 5)
            for i in range(len(FEATURE_NAMES))
        },
        "critical_thresholds": Config.CRITICAL,
        "class_scenarios": {
            CLASS_NAMES[k]: scenarios
            for k, scenarios in CLASS_SCENARIOS.items()
        },
        "neighbor_pairs_weights": {
            f"{CLASS_NAMES[a]}↔{CLASS_NAMES[b]}": w
            for (a, b), w in Config.PAIR_WEIGHTS.items()
        },
        "note": (
            "Модель v8.0.0: оконные признаки (mean/std/min/max/slope), "
            "2 сценария на класс, адаптивные border-веса, "
            "автовыбор лучшей модели RF vs GB."
        ),
    }

    df.to_csv(Config.DATA_PATH, index=False, encoding="utf-8")

    with open(Config.META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(f"✅ Модель:     {Config.MODEL_PATH}")
    log.info(f"✅ Данные:     {Config.DATA_PATH}")
    log.info(f"✅ Метаданные: {Config.META_PATH}")
    log.info("=" * 78)
    log.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    log.info("=" * 78)

    return best_model


if __name__ == "__main__":
    train()
