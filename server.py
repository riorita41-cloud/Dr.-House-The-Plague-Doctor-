# server.py — ФИНАЛЬНАЯ ВЕРСИЯ (20 признаков V8)
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from datetime import datetime
from collections import deque
import joblib
import numpy as np
import json
import logging
import subprocess
import threading
import io
import sys
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Загружаем модель и метаданные
model = joblib.load('doctor_house_model_v8.pkl')
with open('doctor_house_metadata_v8.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

CLASS_NAMES = meta['classes']
FEATURE_NAMES = meta['features']
WINDOW_SIZE = meta['window_size']
logger.info(f"Модель: {len(CLASS_NAMES)} классов, {len(FEATURE_NAMES)} признаков, окно: {WINDOW_SIZE}")

# База знаний (подробные рецепты)
STATE_DB = {
    "Норма": {
        "name": "🟢 НОРМА",
        "icon": "🟢",
        "color": "#2ecc71",
        "advice": "Поддерживайте режим сна (7-9 часов).\nРегулярная физическая активность 30 мин/день.\nСбалансированное питание и гидратация.",
        "recipe": (
            "╔══════════════════════════════════════╗\n"
            "║   🧪 СНАДОБЬЯ ЧУМНОГО ДОКТОРА      ║\n"
            "╚══════════════════════════════════════╝\n\n"
            "ПРОФИЛАКТИКА (по желанию):\n\n"
            "🍊 Витамин C — 500 мг 1 раз/день, 7-14 дней.\n"
            "☀️ Витамин D3 — 2000 МЕ/день (по анализу).\n"
            "⚡ Магний B6 — 1 таб 1 раз/день, 2-4 недели.\n\n"
            "ЛЕКАРСТВЕННЫЕ ПРЕПАРАТЫ НЕ ТРЕБУЮТСЯ!"
        ),
        "doctor": "Плановый осмотр: 1 раз в 6 месяцев.",
        "ambulance": ""
    },
    "Напряжение": {
        "name": "🟡 НАПРЯЖЕНИЕ",
        "icon": "🟡",
        "color": "#f1c40f",
        "advice": "Микро-перерывы каждые 60-90 минут.\nДыхание 4-6 (вдох 4 сек, выдох 6 сек) — 10-15 циклов.\nРасслабить челюсть, плечи, кисти.\nГидратация: 150-250 мл воды каждые 1-2 часа.",
        "recipe": (
            "╔══════════════════════════════════════╗\n"
            "║   🧪 СНАДОБЬЯ ЧУМНОГО ДОКТОРА      ║\n"
            "╚══════════════════════════════════════╝\n\n"
            "ЛЕКАРСТВЕННЫЕ ПРЕПАРАТЫ НЕ ТРЕБУЮТСЯ.\n\n"
            "✨ Достаточно немедикаментозных мер:\n\n"
            "• Дыхательная гимнастика 4-6.\n"
            "• Микро-перерывы каждые 60-90 мин.\n"
            "• Лёгкая растяжка шеи, плеч, предплечий.\n"
            "• Сон 7-9 часов, минимум экранов за 1 час до сна."
        ),
        "doctor": "Если напряжение ежедневно > 7-10 дней — терапевт.",
        "ambulance": ""
    },
    "Утомление": {
        "name": "🔵 УТОМЛЕНИЕ",
        "icon": "🔵",
        "color": "#3498db",
        "advice": "Немедленный отдых 15-20 минут.\nСтакан воды + лёгкий перекус (банан, орехи, йогурт).\n10 глубоких вдохов (вдох 4 сек, выдох 6 сек).\nЛёгкий самомассаж шеи, плеч и кистей рук.",
        "recipe": (
            "╔══════════════════════════════════════╗\n"
            "║   🧪 СНАДОБЬЯ ЧУМНОГО ДОКТОРА      ║\n"
            "╚══════════════════════════════════════╝\n\n"
            "ВИТАМИНЫ И МИНЕРАЛЫ (безрецептурно):\n\n"
            "⚡ Магний B6 (Магне B6, Магнелис B6)\n"
            "   — 1 таб 2-3 раза/день, курс 1 месяц.\n\n"
            "💊 Глицин\n"
            "   — 1 таб (100 мг) под язык 2-3 раза/день, 2-4 недели.\n\n"
            "😴 Мелатонин\n"
            "   — 1-3 мг за час до сна, 7-14 дней (при нарушениях сна).\n\n"
            "ПО АНАЛИЗАМ (по назначению врача):\n"
            "☀️ Витамин D3 — 2000-4000 МЕ/день.\n"
            "🔬 Витамин B12 — по схеме врача.\n\n"
            "ЕСЛИ УТОМЛЕНИЕ ОТ СТРЕССА:\n"
            "💚 Ново-Пассит — 1 таб 3 раза/день."
        ),
        "doctor": "Терапевт (если часто). Невролог, эндокринолог — по анализам.",
        "ambulance": ""
    },
    "Восстановление": {
        "name": "💜 ВОССТАНОВЛЕНИЕ",
        "icon": "💜",
        "color": "#9b59b6",
        "advice": "Продолжайте дышать спокойно (вдох 4, выдох 6).\nПосидите в тишине 5-10 минут.\nНе хватайтесь сразу за сложные задачи.\nИзбегайте кофе и энергетиков.\nВыйдите на свежий воздух на 5-10 минут.",
        "recipe": (
            "╔══════════════════════════════════════╗\n"
            "║   🧪 СНАДОБЬЯ ЧУМНОГО ДОКТОРА      ║\n"
            "╚══════════════════════════════════════╝\n\n"
            "ДЛЯ УСКОРЕНИЯ ВОССТАНОВЛЕНИЯ:\n\n"
            "⚡ Магний B6 — 1 таб 2 раза/день, 2-4 недели.\n\n"
            "💊 Глицин — 2 таб под язык утром и вечером, 2-4 недели.\n\n"
            "🍊 Витамин C — 500-1000 мг/день, 7-10 дней.\n\n"
            "РАСТИТЕЛЬНЫЕ СРЕДСТВА:\n"
            "🍵 Травяной чай: ромашка, мелисса, мята.\n"
            "🌿 Настойка пустырника — 30 капель на ночь."
        ),
        "doctor": "Врач не требуется при разовом восстановлении.",
        "ambulance": ""
    },
    "Стресс": {
        "name": "🟠 СТРЕСС",
        "icon": "🟠",
        "color": "#e67e22",
        "advice": "⚠️ Дыхание 'квадрат' 4-4-4-4 (5-10 раз).\nУмойтесь холодной водой, выйдите на воздух.\nСожмите/разожмите кулаки 10 раз.\nТехника 5-4-3-2-1 (переключение внимания).",
        "recipe": (
            "╔══════════════════════════════════════╗\n"
            "║   🧪 СНАДОБЬЯ ЧУМНОГО ДОКТОРА      ║\n"
            "╚══════════════════════════════════════╝\n\n"
            "🟢 БЕЗРЕЦЕПТУРНЫЕ ПРЕПАРАТЫ:\n\n"
            "💊 Глицин — 2 таб под язык 2-3 раза/день, 2-4 недели.\n\n"
            "⚡ Магний B6 — 1 таб 2-3 раза/день, 3-4 недели.\n\n"
            "🌿 Валериана — 1-2 таб за час до сна, 2-3 недели.\n\n"
            "🌱 Пустырник — 1 таб 3 раза/день или 30 капель, 2-4 нед.\n\n"
            "💚 Ново-Пассит — 1 таб 3 раза/день, 2-4 недели.\n\n"
            "🔴 РЕЦЕПТУРНЫЕ (только невролог/психиатр!):\n"
            "⚠️ Афобазол — 1 таб 3 раза/день, 2-4 недели.\n"
            "⚠️ Фенибут — курс не более 4-6 недель!\n"
            "⚠️ Адаптол — тревога без сонливости.\n"
            "⚠️ Мексидол — 125 мг 3 раза/день, 2-4 нед."
        ),
        "doctor": "Терапевт, невролог, психотерапевт, кардиолог.",
        "ambulance": "Сильная боль/сжатие в груди — 103!\nДавление > 180/100 — 103!\nПотеря сознания — 103!\nПульс > 120 в покое с одышкой — 103!"
    },
    "Перегрузка": {
        "name": "🔴 ПЕРЕГРУЗКА",
        "icon": "🔴",
        "color": "#e74c3c",
        "advice": "🚨 НЕМЕДЛЕННО прекратите деятельность!\nВыйдите на воздух или в тихое место.\nВыключите телефон и компьютер.\nДыхание: вдох 4 / задержка 4 / выдох 6-8 (10-15 раз).\nУмыться холодной водой, холодное полотенце ко лбу.\nПосидите в полной тишине 10-15 минут.",
        "recipe": (
            "╔══════════════════════════════════════╗\n"
            "║   🧪 СНАДОБЬЯ ЧУМНОГО ДОКТОРА      ║\n"
            "╚══════════════════════════════════════╝\n\n"
            "🟢 ЭКСТРЕННОЕ СНЯТИЕ (безрецептурно, разово):\n\n"
            "💧 Корвалол / Валокордин — 20-30 капель РАЗОВО.\n\n"
            "💊 Глицин — 2-3 таб под язык экстренно.\n\n"
            "🔴 СНИЖЕНИЕ ПУЛЬСА/ДАВЛЕНИЯ (ТОЛЬКО ПО РЕЦЕПТУ!):\n"
            "⚠️ Анаприлин — 10-40 мг по назначению врача.\n"
            "⚠️ Метопролол — 25-100 мг 2 раза/день.\n"
            "⚠️ Бисопролол — 2.5-10 мг 1 раз/день.\n\n"
            "🟡 ВОССТАНОВЛЕНИЕ ПОСЛЕ КРИЗИСА:\n"
            "⚡ Магний B6 — 1 таб 2-3 раза/день, 3-4 недели.\n"
            "💊 Глицин — 2 таб 3 раза/день, 2-4 недели.\n"
            "💚 Ново-Пассит — 1 таб 3 раза/день, 1-2 недели.\n"
            "😴 Мелатонин — 1-3 мг за час до сна, 7-14 дней."
        ),
        "doctor": "Терапевт — в ближайшие 1-2 дня. Невролог, кардиолог.",
        "ambulance": "🚨 Сильная боль в груди — 103!\n🚨 Давление > 160/100 — 103!\n🚨 Потеря сознания — 103!\n🚨 Не можете отдышаться — 103!"
    },
    "Тахикардия": {
        "name": "🟡 ТАХИКАРДИЯ",
        "icon": "🟡",
        "color": "#f1c40f",
        "advice": "Остановить нагрузку, сесть или лечь.\nДыхание 4-6 (вдох 4, выдох 6) — 10-15 циклов.\nУмойтесь холодной водой.\nНЕ пить кофе, энергетики, алкоголь!",
        "recipe": (
            "╔══════════════════════════════════════╗\n"
            "║   🧪 СНАДОБЬЯ ЧУМНОГО ДОКТОРА       ║\n"
            "╚══════════════════════════════════════╝\n\n"
            "🟢 БЕЗРЕЦЕПТУРНЫЕ (для успокоения):\n\n"
            "💧 Корвалол / Валокордин — 20-30 капель РАЗОВО.\n"
            "⚡ Препараты Магния (Магне B6).\n\n"
            "🔴 РЕЦЕПТУРНЫЕ (строго по назначению кардиолога!):\n"
            "⚠️ Бета-блокаторы (Анаприлин, Бисопролол, Метопролол).\n"
            "⚠️ САМОСТОЯТЕЛЬНЫЙ ПРИЁМ ЗАПРЕЩЁН!"
        ),
        "doctor": "Обязательная консультация кардиолога и ЭКГ.",
        "ambulance": "Пульс > 150 в покое, боль в груди, обморок — 103!"
    },
    "Брадикардия": {
        "name": "🟠 БРАДИКАРДИЯ",
        "icon": "🟠",
        "color": "#e67e22",
        "advice": "Избегать резких подъёмов из положения лёжа.\nТёплое питьё (чай).\nЛёгкая разминка, медленная ходьба.",
        "recipe": (
            "╔══════════════════════════════════════╗\n"
            "║   🧪 СНАДОБЬЯ ЧУМНОГО ДОКТОРА      ║\n"
            "╚══════════════════════════════════════╝\n\n"
            "🟢 БЕЗРЕЦЕПТУРНЫЕ:\n"
            "☕ Кофеин-содержащие напитки (чай, кофе) могут временно помочь.\n"
            "Специальных безрецептурных препаратов нет.\n\n"
            "🔴 РЕЦЕПТУРНЫЕ (строго по назначению кардиолога!):\n"
            "⚠️ Атропин, Изадрин и др. применяются в стационаре.\n"
            "В тяжёлых случаях — установка кардиостимулятора."
        ),
        "doctor": "Рекомендуется консультация кардиолога.",
        "ambulance": "Пульс < 40, слабость, обморок — 103!"
    },
    "АСИСТОЛИЯ": {
        "name": "⛔ АСИСТОЛИЯ",
        "icon": "⛔",
        "color": "#c0392b",
        "advice": "1. Проверить сознание (окликнуть, потрясти).\n2. Проверить дыхание (10 сек).\n3. Пульс на сонной артерии (10 сек).\n4. Нет пульса/дыхания → НАЧАТЬ СЛР немедленно!\n   30 компрессий (5-6 см, 100-120/мин) + 2 вдоха.\n5. ВЫЗВАТЬ 103/112 НЕМЕДЛЕННО!",
        "recipe": (
            "╔══════════════════════════════════════╗\n"
            "║   🧪 СНАДОБЬЯ ЧУМНОГО ДОКТОРА      ║\n"
            "╚══════════════════════════════════════╝\n\n"
            "⛔ ПРЕПАРАТЫ ВВОДЯТСЯ ТОЛЬКО РЕАНИМАЦИОННОЙ БРИГАДОЙ!\n\n"
            "(Адреналин, Атропин, Амилорид, Калий-хлорид)\n\n"
            "ДО ПРИЕХАНИЯ ВРАЧЕЙ — ТОЛЬКО НЕПРЕРЫВНЫЙ МАССАЖ СЕРДЦА!"
        ),
        "doctor": "Реаниматолог, кардиолог.",
        "ambulance": "⛔ НЕМЕДЛЕННО 103 или 112!"
    }
}

# Буфер для данных от step3
latest_realtime_data = {
    "pulse": 0, "emg": 0, "gsr": 0, "eeg": 0, "timestamp": None
}

# Скользящее окно для корректного вычисления признаков
window_buffer = deque(maxlen=WINDOW_SIZE)


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


@app.route('/')
def index():
    return send_file('Чумной Доктор.html')


@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.json
        pulse = float(data.get('pulse', 0))
        emg = float(data.get('emg', 0))
        gsr = float(data.get('gsr', 0))
        eeg = float(data.get('eeg', 0))
        
        # Критические проверки
        if pulse <= 0:
            result = STATE_DB["АСИСТОЛИЯ"].copy()
        elif pulse <= 50:
            result = STATE_DB["Брадикардия"].copy()
        elif pulse > 130:
            result = STATE_DB["Тахикардия"].copy()
        else:
            window_buffer.append([pulse, emg, gsr, eeg])
            
            if len(window_buffer) < WINDOW_SIZE:
                padding = WINDOW_SIZE - len(window_buffer)
                pad = [window_buffer[0]] * padding
                win = np.array(list(pad) + list(window_buffer), dtype=float)
            else:
                win = np.array(list(window_buffer), dtype=float)
            
            feats = extract_features(win)
            X = feats.reshape(1, -1)
            prediction = int(model.predict(X)[0])
            probas = model.predict_proba(X)[0]
            confidence = float(probas.max())
            state_name = CLASS_NAMES[prediction]
            result = STATE_DB.get(state_name, STATE_DB["Норма"]).copy()
            result["confidence"] = round(confidence * 100, 1)
            
            # 🔥 Добавляем вероятности всех классов
            result["probabilities"] = {}
            for i, name in enumerate(CLASS_NAMES):
                result["probabilities"][name] = round(float(probas[i]) * 100, 1)
                
        return jsonify(result)
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return jsonify({"name": "❌ ОШИБКА", "color": "#95a5a6", "advice": str(e), "recipe": "Проверьте данные"}), 500


@app.route('/api/realtime', methods=['GET'])
def get_realtime():
    return jsonify(latest_realtime_data)


@app.route('/api/realtime/update', methods=['POST'])
def update_realtime():
    global latest_realtime_data
    data = request.json
    latest_realtime_data = {
        "pulse": float(data.get('pulse', 0)),
        "emg": float(data.get('emg', 0)),
        "gsr": float(data.get('gsr', 0)),
        "eeg": float(data.get('eeg', 0)),
        "timestamp": datetime.now().isoformat()
    }
    return jsonify({"status": "ok"})


# ══════════════════════════════════════════════════════════════
# ADMIN PANEL — управление шагами через UI
# ══════════════════════════════════════════════════════════════

admin_log = []
admin_lock = threading.Lock()
realtime_process = None
realtime_log_path = None


def log_msg(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    with admin_lock:
        admin_log.append(f'[{ts}] {msg}')
        # держим последние 100 строк
        while len(admin_log) > 100:
            admin_log.pop(0)
    logger.info(msg)


@app.route('/api/admin/status')
def admin_status():
    model_exists = os.path.exists('doctor_house_model_v8.pkl')
    data_exists = os.path.exists('training_data_v8.csv')
    meta_exists = os.path.exists('doctor_house_metadata_v8.json')
    
    status = {
        'model_loaded': model_exists,
        'model_accuracy': meta.get('quality', {}).get('best_test_accuracy', 0) if meta_exists else 0,
        'data_exists': data_exists,
        'window_size': WINDOW_SIZE,
        'classes': CLASS_NAMES,
        'features': len(FEATURE_NAMES),
        'realtime_running': realtime_process is not None and realtime_process.poll() is None,
        'server_uptime': str(datetime.now() - globals().get('_start_time', datetime.now())).split('.')[0],
    }
    return jsonify(status)


@app.route('/api/admin/logs')
def admin_get_logs():
    with admin_lock:
        log_content = '\n'.join(admin_log)
    return jsonify({'logs': log_content})


def run_train():
    log_msg('🚀 Запуск обучения модели...')
    try:
        result = subprocess.run(
            [sys.executable, 'step1_train_v8_final.py'],
            capture_output=True, text=True, timeout=600
        )
        log_msg(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            log_msg(f'❌ Ошибка: {result.stderr[-500:]}')
        else:
            log_msg('✅ Обучение завершено успешно!')
            # перезагружаем модель
            globals()['model'] = joblib.load('doctor_house_model_v8.pkl')
    except subprocess.TimeoutExpired:
        log_msg('❌ Обучение прервано по таймауту (600 сек)')
    except Exception as e:
        log_msg(f'❌ Ошибка: {e}')


def run_test():
    log_msg('🧪 Запуск тестирования модели...')
    try:
        result = subprocess.run(
            [sys.executable, 'step2_test_v8_full_report.py'],
            capture_output=True, text=True, timeout=120
        )
        log_msg(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            log_msg(f'❌ Ошибка: {result.stderr[-500:]}')
        else:
            log_msg('✅ Тестирование завершено!')
    except subprocess.TimeoutExpired:
        log_msg('❌ Тестирование прервано по таймауту (120 сек)')
    except Exception as e:
        log_msg(f'❌ Ошибка: {e}')


@app.route('/api/admin/train', methods=['POST'])
def admin_train():
    if not os.path.exists('training_data_v8.csv'):
        return jsonify({'error': 'Файл training_data_v8.csv не найден'}), 400
    thread = threading.Thread(target=run_train, daemon=True)
    thread.start()
    log_msg('📦 Обучение запущено в фоне')
    return jsonify({'status': 'started', 'message': 'Обучение запущено'})


@app.route('/api/admin/test', methods=['POST'])
def admin_test():
    thread = threading.Thread(target=run_test, daemon=True)
    thread.start()
    log_msg('🧪 Тестирование запущено в фоне')
    return jsonify({'status': 'started', 'message': 'Тестирование запущено'})


@app.route('/api/admin/realtime/start', methods=['POST'])
def admin_realtime_start():
    global realtime_process, realtime_log_path
    if realtime_process is not None and realtime_process.poll() is None:
        return jsonify({'error': 'Режим реального времени уже запущен'}), 400
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    realtime_log_path = f"admin_realtime_{ts}.log"
    realtime_process = subprocess.Popen(
        [sys.executable, 'step3_realtime_v8_2.py'],
        stdout=open(realtime_log_path, 'w'),
        stderr=subprocess.STDOUT,
    )
    log_msg(f'📡 Режим реального времени запущен (PID: {realtime_process.pid})')
    return jsonify({'status': 'started', 'pid': realtime_process.pid})


@app.route('/api/admin/realtime/stop', methods=['POST'])
def admin_realtime_stop():
    global realtime_process
    if realtime_process is None or realtime_process.poll() is not None:
        return jsonify({'error': 'Режим реального времени не запущен'}), 400
    realtime_process.terminate()
    realtime_process.wait(timeout=5)
    log_msg('⏹ Режим реального времени остановлен')
    realtime_process = None
    return jsonify({'status': 'stopped'})


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response


if __name__ == '__main__':
    globals()['_start_time'] = datetime.now()
    print("=" * 60)
    print("🌐 ДОКТОР ХАУС — Сервер v8.2 FINAL")
    print(f"   Адрес: http://127.0.0.1:5000")
    print(f"   Признаков: {len(FEATURE_NAMES)} (окно {WINDOW_SIZE} точек)")
    print(f"   Классов: {len(CLASS_NAMES)}")
    print("=" * 60)
    app.run(debug=True, port=5000)