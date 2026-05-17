# step3_realtime_v8_2.py — ДОКТОР ХАУС (реальное время + подробные рецепты)
import requests
import socket
import os
import csv
import json
import time
import numpy as np
import joblib
from datetime import datetime
from collections import deque


# ═══════════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════════

class Config:
    UDP_IP        = "0.0.0.0"
    UDP_PORT      = 5005
    BUFFER_SIZE   = 1024
    ANALYZE_EVERY = 1.0
    LOG_FOLDER    = "logs_realtime"
    MODEL_PATH    = "doctor_house_model_v8.pkl"
    META_PATH     = "doctor_house_metadata_v8.json"
    CLEAR_CONSOLE        = False
    HISTORY_SIZE         = 5
    NO_DATA_NOTIFY_EVERY = 5.0


ESP_PACKET_SIZE = 11
RAW_CHANNELS    = ["пульс", "эмг", "кгр", "ээг"]


# ═══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════

def load_model():
    print("=" * 68)
    print("   ДОКТОР ХАУС — Реальное время v3.2.1")
    print("=" * 68)
    print()
    try:
        model = joblib.load(Config.MODEL_PATH)
        with open(Config.META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"✅ Модель:    {Config.MODEL_PATH}")
        print(f"✅ Версия:    {meta.get('version', '?')}")
        print(f"✅ Признаков: {len(meta['features'])}")
        print(f"✅ Классов:   {len(meta['classes'])}")
        print(f"✅ Окно:      {meta['window_size']} точек")
        print(f"✅ Порог:     {meta['confidence_threshold']:.0%}")
        print()
        return model, meta
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        raise SystemExit


# ═══════════════════════════════════════════════════════════════════
# ДЕКОДИРОВАНИЕ ПАКЕТА ESP32
# ═══════════════════════════════════════════════════════════════════

def decode_esp_packet(data: bytes):
    if len(data) != ESP_PACKET_SIZE:
        return None
    if data[0] != 0xAA or data[10] != 0x55:
        return None
    crc = 0
    for i in range(1, 9):
        crc ^= data[i]
    if crc != data[9]:
        return None
    pulse = (data[1] << 8) | data[2]
    emg   = (data[3] << 8) | data[4]
    gsr   = (data[5] << 8) | data[6]
    eeg   = (data[7] << 8) | data[8]
    return pulse, emg, gsr, eeg


def raw_to_medical(pulse, emg, gsr, eeg):
    # Кусочно-линейная интерполяция через опорные точки
    # ТОЧНО: 0→0, 1000→40, 2400→85, 4000→200
    x = pulse
    
    if x <= 0:
        ps = 0
    elif x <= 1000:
        # 0..40 уд/мин
        ps = 40 * x / 1000
    elif x <= 2400:
        # 40..85 уд/мин
        ps = 40 + 45 * (x - 1000) / 1400
    elif x <= 4000:
        # 85..200 уд/мин
        ps = 85 + 115 * (x - 2400) / 1600
    else:
        ps = 200
    
    ps = round(float(ps), 1)
    
    emg_pct = round(((emg / 4095.0) ** 0.7) * 100, 1)
    gsr_pct = round((gsr / 4095.0) * 100, 1)
    eeg_pct = round(((eeg / 4095.0) ** 0.8) * 100, 1)
    
    return ps, emg_pct, gsr_pct, eeg_pct
# ═══════════════════════════════════════════════════════════════════
# КРИТИЧЕСКИЙ ДЕТЕКТОР
# ═══════════════════════════════════════════════════════════════════

class CriticalDetector:
    def __init__(self, cfg: dict):
        self.asystole    = cfg["asystole_pulse"]
        self.bradycardia = cfg["bradycardia_pulse"]
        self.tachycardia = cfg["tachycardia_pulse"]

    def check(self, pulse_bpm: float) -> dict:
        if pulse_bpm <= self.asystole:
            return {"code": "asystole", "name": "АСИСТОЛИЯ", "severity": "critical",
                    "message": f"Пульс = {pulse_bpm:.0f} уд/мин! НЕМЕДЛЕННО вызовите скорую (103)!"}
        if pulse_bpm <= self.bradycardia:
            return {"code": "bradycardia", "name": "Брадикардия", "severity": "high",
                    "message": f"Пульс {pulse_bpm:.0f} ≤ {self.bradycardia} уд/мин. Рекомендуется консультация кардиолога."}
        if pulse_bpm > self.tachycardia:
            return {"code": "tachycardia", "name": "Тахикардия", "severity": "high",
                    "message": f"Пульс {pulse_bpm:.0f} > {self.tachycardia} уд/мин. Рекомендуется ЭКГ и консультация кардиолога."}
        return {"code": "normal", "name": "норма", "severity": "none", "message": None}


# ═══════════════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ═══════════════════════════════════════════════════════════════════

def extract_features(window: np.ndarray) -> np.ndarray:
    features = []
    n = window.shape[0]
    x = np.arange(n, dtype=float)
    for ch_idx in range(window.shape[1]):
        col = window[:, ch_idx]
        features.append(float(np.mean(col)))
        features.append(float(np.std(col)))
        features.append(float(np.min(col)))
        features.append(float(np.max(col)))
        slope = float(np.polyfit(x, col, 1)[0]) if n > 1 else 0.0
        features.append(slope)
    return np.array(features, dtype=float)


# ═══════════════════════════════════════════════════════════════════
# СКОЛЬЗЯЩЕЕ ОКНО
# ═══════════════════════════════════════════════════════════════════

class SlidingWindow:
    def __init__(self, size: int):
        self.size   = size
        self.buffer = deque(maxlen=size)

    def add(self, ps: float, emg: float, gsr: float, eeg: float):
        self.buffer.append([ps, emg, gsr, eeg])

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.size

    def get_window(self) -> np.ndarray:
        return np.array(list(self.buffer), dtype=float)

    def get_means(self) -> dict:
        arr = self.get_window()
        return {
            "пульс": float(np.mean(arr[:, 0])),
            "эмг":   float(np.mean(arr[:, 1])),
            "кгр":   float(np.mean(arr[:, 2])),
            "ээг":   float(np.mean(arr[:, 3])),
        }


# ═══════════════════════════════════════════════════════════════════
# БАЗА СОСТОЯНИЙ (подробные рецепты из step2)
# ═══════════════════════════════════════════════════════════════════

STATE_DB = {
    "Норма": {
        "icon": "🟢",
        "advice": [
            "Всё в порядке. Поддерживайте режим сна и активности.",
            "Регулярная физическая активность 30 мин/день.",
            "Сбалансированное питание и гидратация.",
        ],
        "recipe": [
            "ЛЕКАРСТВЕННЫЕ ПРЕПАРАТЫ НЕ ТРЕБУЮТСЯ.",
            "ПРОФИЛАКТИКА (по желанию):",
            "  • Витамин C — 500 мг 1 раз/день, 7-14 дней.",
            "  • Витамин D3 — 2000 МЕ/день (по анализу).",
            "  • Магний B6 — 1 таб 1 раз/день, 2-4 недели.",
        ],
        "doctor": "Плановый осмотр: 1 раз в 6 месяцев.",
        "ambulance": [],
    },
    "Напряжение": {
        "icon": "🟡",
        "advice": [
            "Микро-перерывы каждые 60-90 минут.",
            "Дыхание 4-6 (вдох 4 сек, выдох 6 сек) — 10-15 циклов.",
            "Расслабить челюсть, плечи, кисти.",
            "Гидратация: 150-250 мл воды каждые 1-2 часа.",
        ],
        "recipe": [
            "ЛЕКАРСТВЕННЫЕ ПРЕПАРАТЫ НЕ ТРЕБУЮТСЯ.",
            "НЕМЕДИКАМЕНТОЗНЫЕ МЕРЫ:",
            "  • Дыхательная гимнастика 4-6.",
            "  • Микро-перерывы каждые 60-90 мин.",
            "  • Лёгкая растяжка шеи, плеч, предплечий.",
            "  • Сон 7-9 часов, минимум экранов за 1 час до сна.",
        ],
        "doctor": "Если напряжение ежедневно > 7-10 дней — терапевт.",
        "ambulance": [],
    },
    "Утомление": {
        "icon": "🔵",
        "advice": [
            "Немедленный отдых 15-20 минут.",
            "Стакан воды + лёгкий перекус (банан, орехи, йогурт).",
            "10 глубоких вдохов (вдох 4 сек, выдох 6 сек).",
            "Лёгкий самомассаж шеи, плеч и кистей рук.",
        ],
        "recipe": [
            "ВИТАМИНЫ И МИНЕРАЛЫ (безрецептурно):",
            "  • Магний B6 — 1 таб 2-3 раза/день, курс 1 месяц.",
            "  • Комплекс витаминов B (Нейромультивит) — 1 таб 1 раз/день, 30 дней.",
            "  • Глицин — 1 таб (100 мг) под язык 2-3 раза/день, 2-4 недели.",
            "ПО АНАЛИЗАМ (по назначению врача):",
            "  • Витамин D3 — 2000-4000 МЕ/день.",
            "  • Витамин B12 — по схеме врача.",
            "ЕСЛИ УТОМЛЕНИЕ ОТ СТРЕССА:",
            "  • Ново-Пассит — 1 таб 3 раза/день.",
            "  • Мелатонин — 1-3 мг за час до сна, 7-14 дней.",
        ],
        "doctor": "Терапевт (если часто). Невролог, эндокринолог — по анализам.",
        "ambulance": [],
    },
    "Восстановление": {
        "icon": "💜",
        "advice": [
            "Хорошее состояние. Продолжайте отдыхать.",
            "Не хватайтесь сразу за сложные задачи.",
            "Избегайте кофе и энергетиков.",
            "Выйдите на свежий воздух на 5-10 минут.",
        ],
        "recipe": [
            "ДЛЯ УСКОРЕНИЯ ВОССТАНОВЛЕНИЯ:",
            "  • Магний B6 — 1 таб 2 раза/день, 2-4 недели.",
            "  • Глицин — 2 таб под язык утром и вечером, 2-4 недели.",
            "  • Витамин C — 500-1000 мг/день, 7-10 дней.",
            "РАСТИТЕЛЬНЫЕ СРЕДСТВА:",
            "  • Травяной чай: ромашка, мелисса, мята.",
            "  • Настойка пустырника — 30 капель на ночь.",
        ],
        "doctor": "Врач не требуется при разовом восстановлении.",
        "ambulance": [],
    },
    "Стресс": {
        "icon": "🟠",
        "advice": [
            "⚠️ Дыхание 'квадрат' 4-4-4-4 (5-10 раз).",
            "Умойтесь холодной водой, выйдите на воздух.",
            "Сожмите/разожмите кулаки 10 раз.",
            "Техника 5-4-3-2-1 (переключение внимания).",
        ],
        "recipe": [
            "БЕЗРЕЦЕПТУРНЫЕ ПРЕПАРАТЫ:",
            "  • Глицин — 2 таб под язык 2-3 раза/день, 2-4 недели.",
            "  • Магний B6 — 1 таб 2-3 раза/день, 3-4 недели.",
            "  • Валериана — 1-2 таб за час до сна, 2-3 недели.",
            "  • Пустырник — 1 таб 3 раза/день или 30 капель, 2-4 нед.",
            "  • Ново-Пассит — 1 таб 3 раза/день, 2-4 недели.",
            "РЕЦЕПТУРНЫЕ (только невролог/психиатр):",
            "  • Афобазол, Фенибут, Адаптол, Мексидол.",
        ],
        "doctor": "Терапевт, невролог, психотерапевт, кардиолог.",
        "ambulance": [
            "Сильная боль/сжатие в груди — 103!",
            "Давление > 180/100 — 103!",
            "Потеря сознания — 103!",
            "Пульс > 120 в покое с одышкой — 103!",
        ],
    },
    "Перегрузка": {
        "icon": "🔴",
        "advice": [
            "🚨 НЕМЕДЛЕННО прекратите деятельность!",
            "Выйдите на воздух или в тихое место.",
            "Выключите телефон и компьютер.",
            "Дыхание: вдох 4 / задержка 4 / выдох 6-8 (10-15 раз).",
            "Умыться холодной водой, холодное полотенце ко лбу.",
            "Посидите в полной тишине 10-15 минут.",
        ],
        "recipe": [
            "ЭКСТРЕННОЕ СНЯТИЕ (безрецептурно, разово):",
            "  • Корвалол / Валокордин — 20-30 капель РАЗОВО.",
            "  • Глицин — 2-3 таб под язык экстренно.",
            "СНИЖЕНИЕ ПУЛЬСА/ДАВЛЕНИЯ (ТОЛЬКО РЕЦЕПТ!):",
            "  • Анаприлин, Метопролол, Бисопролол.",
            "ВОССТАНОВЛЕНИЕ ПОСЛЕ КРИЗИСА:",
            "  • Магний B6, Глицин, Ново-Пассит, Мелатонин.",
        ],
        "doctor": "Терапевт — в ближайшие 1-2 дня. Невролог, кардиолог.",
        "ambulance": [
            "🚨 Сильная боль в груди — 103!",
            "🚨 Давление > 160/100 — 103!",
            "🚨 Потеря сознания — 103!",
            "🚨 Не можете отдышаться — 103!",
        ],
    },
    "Тахикардия": {
        "icon": "🟡",
        "advice": [
            "Остановить нагрузку, сесть или лечь.",
            "Дыхание 4-6 (вдох 4, выдох 6) — 10-15 циклов.",
            "Умойтесь холодной водой.",
            "НЕ пить кофе, энергетики, алкоголь!",
        ],
        "recipe": ["Консультация кардиолога. ЭКГ, Холтер-мониторинг."],
        "doctor": "Кардиолог.",
        "ambulance": ["Пульс > 150 в покое, боль в груди, обморок — 103!"],
    },
    "Брадикардия": {
        "icon": "🟠",
        "advice": [
            "Избегать резких подъёмов из положения лёжа.",
            "Тёплое питьё (чай).",
            "Лёгкая разминка, медленная ходьба.",
        ],
        "recipe": ["Консультация кардиолога. ЭКГ."],
        "doctor": "Кардиолог.",
        "ambulance": ["Пульс < 40, слабость, обморок — 103!"],
    },
    "АСИСТОЛИЯ": {
        "icon": "⛔",
        "advice": [
            "1. Проверить сознание (окликнуть, потрясти).",
            "2. Проверить дыхание (10 сек).",
            "3. Пульс на сонной артерии (10 сек).",
            "4. Нет пульса/дыхания → НАЧАТЬ СЛР немедленно!",
            "   30 компрессий (5-6 см, 100-120/мин) + 2 вдоха.",
            "5. ВЫЗВАТЬ 103/112 НЕМЕДЛЕННО!",
        ],
        "recipe": ["РЕАНИМАЦИЯ! СЛР до прибытия скорой!"],
        "doctor": "Реаниматолог, кардиолог.",
        "ambulance": ["⛔ НЕМЕДЛЕННО 103 или 112!"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# ВЫВОД ОТЧЁТА
# ═══════════════════════════════════════════════════════════════════

class StateHistory:
    def __init__(self, size: int):
        self.records = deque(maxlen=size)

    def add(self, ts: str, clean_state: str, confidence: float):
        self.records.append((ts, clean_state, confidence))

    def print_history(self):
        if not self.records:
            return
        print("\n  🕐 ИСТОРИЯ (последние состояния):")
        for ts, state, conf in self.records:
            icon = STATE_DB.get(state, {}).get("icon", "❓")
            conf_str = f"{conf:.0%}" if conf > 0 else "—"
            print(f"     {ts}  {icon} {state:<18} {conf_str}")


def print_report(means, state_name, clean_state, probas, critical, class_names, conf_thresh, packet_num, error_count, bad_size, ts, history):
    if Config.CLEAR_CONSOLE:
        os.system("cls" if os.name == "nt" else "clear")

    db = STATE_DB.get(clean_state, STATE_DB["Норма"])
    icon = db["icon"]

    print("\n" + "=" * 68)
    print(f"   ДОКТОР ХАУС  |  {ts}  |  #{packet_num}  |  CRC: {error_count}  |  Плохих: {bad_size}")
    print("=" * 68)

    print("\n  📊 ПОКАЗАТЕЛИ (среднее за окно):")
    print("  ┌──────────────────────────────────────────────────┐")
    print(f"  │  Пульс:  {means['пульс']:6.1f} уд/мин                    │")
    print(f"  │  ЭМГ:    {means['эмг']:6.1f} %                         │")
    print(f"  │  КГР:    {means['кгр']:6.1f} %                         │")
    print(f"  │  ЭЭГ:    {means['ээг']:6.1f} %                         │")
    print("  └──────────────────────────────────────────────────┘")

    if critical["code"] != "normal":
        print(f"\n  {'!' * 52}")
        print(f"  {icon}  {clean_state}")
        if critical["message"]:
            print(f"  {critical['message']}")
        print(f"  {'!' * 52}")

    if probas is not None:
        max_p = float(probas.max())
        print()
        if max_p < conf_thresh:
            print(f"  ⚠️  СОСТОЯНИЕ: НЕОПРЕДЕЛЁННО ({clean_state})  {max_p:.0%}")
        else:
            mark = "✅" if max_p >= 0.75 else "🔶"
            print(f"  {mark} СОСТОЯНИЕ: {icon} {clean_state} ({max_p:.0%})")

        print("\n  📈 Вероятности:")
        top3 = np.argsort(probas)[::-1][:3]
        for idx in top3:
            p = float(probas[idx])
            bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
            print(f"     {class_names[idx]:18} [{bar}] {p*100:5.1f}%")

    advice = db.get("advice", [])
    if advice:
        print("\n  💡 РЕКОМЕНДАЦИИ:")
        for line in advice:
            print(f"     {line}")

    recipe = db.get("recipe", [])
    if recipe:
        print("\n  💊 РЕЦЕПТ:")
        for line in recipe:
            print(f"     {line}")

    doctor = db.get("doctor", "")
    if doctor:
        print(f"\n  👨‍⚕️ ВРАЧ: {doctor}")

    ambulance = db.get("ambulance", [])
    if ambulance:
        print("\n  🚑 СКОРАЯ:")
        for line in ambulance:
            print(f"     {line}")

    history.print_history()
    print("\n  [Ctrl+C] — остановить")
    print("=" * 68)


def send_to_html_server(means, state_name, confidence):
    try:
        requests.post(
            'http://127.0.0.1:5000/api/realtime/update',
            json={
                "pulse": round(means["пульс"], 1),
                "emg": round(means["эмг"], 1),
                "gsr": round(means["кгр"], 1),
                "eeg": round(means["ээг"], 1),
            },
            timeout=0.5
        )
    except:
        pass


# ═══════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ В CSV
# ═══════════════════════════════════════════════════════════════════

def init_result_log(path: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Время", "Пакет", "Пульс_mean", "ЭМГ_mean", "КГР_mean", "ЭЭГ_mean", "Состояние", "Уверенность_%", "Критическое"])


def write_result_log(path, ts, packet_num, means, state_name, confidence, critical):
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([ts, packet_num, round(means["пульс"], 1), round(means["эмг"], 1), round(means["кгр"], 1), round(means["ээг"], 1), state_name, round(confidence * 100, 1), critical["code"]])


# ═══════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ЦИКЛ
# ═══════════════════════════════════════════════════════════════════

def main():
    model, meta = load_model()
    class_names = meta["classes"]
    conf_thresh = meta["confidence_threshold"]
    window_size = meta["window_size"]

    detector = CriticalDetector(meta["critical_thresholds"])
    window   = SlidingWindow(window_size)
    history  = StateHistory(Config.HISTORY_SIZE)

    ts_start   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_log = os.path.join(Config.LOG_FOLDER, f"results_{ts_start}.csv")
    init_result_log(result_log)
    print(f"📄 Лог результатов: {result_log}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((Config.UDP_IP, Config.UDP_PORT))
    sock.settimeout(1.0)

    print(f"📡 Ожидание ESP32 на UDP:{Config.UDP_PORT}...")
    print(f"   Окно: {window_size} пакетов | Анализ каждые {Config.ANALYZE_EVERY} сек")
    print("   Нажмите Ctrl+C для остановки")
    print("-" * 68)

    packet_count = 0
    error_count = 0
    bad_size_count = 0
    last_analyze = time.time()
    last_no_data = 0.0

    try:
        while True:
            try:
                data, _ = sock.recvfrom(Config.BUFFER_SIZE)
            except socket.timeout:
                now = time.time()
                if now - last_no_data >= Config.NO_DATA_NOTIFY_EVERY:
                    print("⏳ Нет данных от ESP32...")
                    last_no_data = now
                continue

            if len(data) != ESP_PACKET_SIZE:
                bad_size_count += 1
                continue

            decoded = decode_esp_packet(data)
            if not decoded:
                error_count += 1
                continue

            packet_count += 1
            ps, emg_pct, gsr_pct, eeg_pct = raw_to_medical(*decoded)
            window.add(ps, emg_pct, gsr_pct, eeg_pct)

            if not window.is_ready():
                filled = len(window.buffer)
                if filled in (window_size // 4, window_size // 2, window_size * 3 // 4, window_size):
                    pct = int(filled / window_size * 20)
                    bar = "█" * pct + "░" * (20 - pct)
                    print(f"  Накопление: [{bar}] {filled}/{window_size}  PS={ps:.1f} EMG={emg_pct:.1f}% GSR={gsr_pct:.1f}% EEG={eeg_pct:.1f}%")
                continue

            now = time.time()
            if now - last_analyze < Config.ANALYZE_EVERY:
                continue
            last_analyze = now

            ts = datetime.now().strftime("%H:%M:%S")
            win_arr = window.get_window()
            means = window.get_means()

            critical = detector.check(means["пульс"])

            if critical["code"] != "normal":
                clean_state = critical["name"]
                state_name = clean_state
                confidence = 1.0
                probas = None
            else:
                feats = extract_features(win_arr)
                X = feats.reshape(1, -1)
                probas = model.predict_proba(X)[0]
                pred = int(model.predict(X)[0])
                max_p = float(probas.max())
                clean_state = class_names[pred]
                state_name = clean_state if max_p >= conf_thresh else f"НЕОПРЕДЕЛЁННО ({clean_state})"
                confidence = max_p

            write_result_log(result_log, ts, packet_count, means, state_name, confidence, critical)
            send_to_html_server(means, state_name, confidence)
            history.add(ts, clean_state, confidence)

            print_report(means, state_name, clean_state, probas, critical, class_names, conf_thresh, packet_count, error_count, bad_size_count, ts, history)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 68)
        print("   ОСТАНОВЛЕНО")
        print("=" * 68)
        print(f"📦 Всего пакетов:    {packet_count}")
        print(f"⚠️  Ошибок CRC:       {error_count}")
        print(f"⚠️  Битых по размеру: {bad_size_count}")
        print(f"💾 Лог:              {result_log}")
        print("=" * 68)
    finally:
        sock.close()


if __name__ == "__main__":
    main()