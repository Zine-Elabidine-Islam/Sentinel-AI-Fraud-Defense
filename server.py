import asyncio
import csv
import io
import json
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf

app = FastAPI(title="SentinelAI - Live Fraud Defense")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

model = tf.keras.models.load_model(str(BASE_DIR / "fraud_model.keras"))
scaler = joblib.load(str(BASE_DIR / "scaler.pkl"))
feature_cols = joblib.load(str(BASE_DIR / "feature_columns.pkl"))

DZD_PER_USD = 134.50

wilayas = {
    "01 - Adrar": {"lat": 27.8742, "lon": -0.2939, "ip_prefix": "105.101"},
    "02 - Chlef": {"lat": 36.1654, "lon": 1.3345, "ip_prefix": "41.101"},
    "03 - Laghouat": {"lat": 33.8000, "lon": 2.8651, "ip_prefix": "105.103"},
    "04 - Oum El Bouaghi": {"lat": 35.8754, "lon": 7.1135, "ip_prefix": "197.201"},
    "05 - Batna": {"lat": 35.5558, "lon": 6.1741, "ip_prefix": "105.105"},
    "06 - Bejaia": {"lat": 36.7559, "lon": 5.0843, "ip_prefix": "197.200"},
    "07 - Biskra": {"lat": 34.8500, "lon": 5.7333, "ip_prefix": "197.204"},
    "08 - Bechar": {"lat": 31.6167, "lon": -2.2167, "ip_prefix": "41.108"},
    "09 - Blida": {"lat": 36.4700, "lon": 2.8300, "ip_prefix": "41.107"},
    "10 - Bouira": {"lat": 36.3749, "lon": 3.9020, "ip_prefix": "105.110"},
    "11 - Tamanrasset": {"lat": 22.7850, "lon": 5.5228, "ip_prefix": "197.211"},
    "12 - Tebessa": {"lat": 35.4042, "lon": 8.1242, "ip_prefix": "197.212"},
    "13 - Tlemcen": {"lat": 34.8783, "lon": -1.3150, "ip_prefix": "41.109"},
    "14 - Tiaret": {"lat": 35.3710, "lon": 1.3170, "ip_prefix": "41.114"},
    "15 - Tizi Ouzou": {"lat": 36.7118, "lon": 4.0459, "ip_prefix": "197.206"},
    "16 - Algiers": {"lat": 36.7538, "lon": 3.0588, "ip_prefix": "105.101"},
    "17 - Djelfa": {"lat": 34.6728, "lon": 3.2630, "ip_prefix": "105.117"},
    "18 - Jijel": {"lat": 36.8205, "lon": 5.7667, "ip_prefix": "197.202"},
    "19 - Setif": {"lat": 36.1911, "lon": 5.4137, "ip_prefix": "105.111"},
    "20 - Saida": {"lat": 34.8303, "lon": 0.1517, "ip_prefix": "41.120"},
    "21 - Skikda": {"lat": 36.8764, "lon": 6.9092, "ip_prefix": "197.201"},
    "22 - Sidi Bel Abbes": {"lat": 35.1899, "lon": -0.6308, "ip_prefix": "41.105"},
    "23 - Annaba": {"lat": 36.9000, "lon": 7.7667, "ip_prefix": "105.109"},
    "24 - Guelma": {"lat": 36.4621, "lon": 7.4261, "ip_prefix": "197.224"},
    "25 - Constantine": {"lat": 36.3650, "lon": 6.6147, "ip_prefix": "105.106"},
    "26 - Medea": {"lat": 36.2642, "lon": 2.7539, "ip_prefix": "41.108"},
    "27 - Mostaganem": {"lat": 35.9311, "lon": 0.0892, "ip_prefix": "41.103"},
    "28 - M'Sila": {"lat": 35.7058, "lon": 4.5419, "ip_prefix": "105.128"},
    "29 - Mascara": {"lat": 35.3966, "lon": 0.1403, "ip_prefix": "41.129"},
    "30 - Ouargla": {"lat": 31.9500, "lon": 5.3167, "ip_prefix": "197.210"},
    "31 - Oran": {"lat": 35.6971, "lon": -0.6308, "ip_prefix": "105.104"},
    "32 - El Bayadh": {"lat": 33.6832, "lon": 1.0193, "ip_prefix": "41.132"},
    "33 - Illizi": {"lat": 26.4833, "lon": 8.4667, "ip_prefix": "197.233"},
    "34 - Bordj Bou Arreridj": {"lat": 36.0732, "lon": 4.7611, "ip_prefix": "105.134"},
    "35 - Boumerdes": {"lat": 36.7664, "lon": 3.4772, "ip_prefix": "105.102"},
    "36 - El Tarf": {"lat": 36.7672, "lon": 8.3138, "ip_prefix": "197.236"},
    "37 - Tindouf": {"lat": 27.6711, "lon": -8.1474, "ip_prefix": "41.137"},
    "38 - Tissemsilt": {"lat": 35.6072, "lon": 1.8108, "ip_prefix": "41.138"},
    "39 - El Oued": {"lat": 33.3561, "lon": 6.8631, "ip_prefix": "197.239"},
    "40 - Khenchela": {"lat": 35.4358, "lon": 7.1433, "ip_prefix": "197.240"},
    "41 - Souk Ahras": {"lat": 36.2864, "lon": 7.9511, "ip_prefix": "197.241"},
    "42 - Tipaza": {"lat": 36.5897, "lon": 2.4475, "ip_prefix": "105.103"},
    "43 - Mila": {"lat": 36.4503, "lon": 6.2644, "ip_prefix": "197.243"},
    "44 - Ain Defla": {"lat": 36.2641, "lon": 1.9679, "ip_prefix": "41.144"},
    "45 - Naama": {"lat": 33.2667, "lon": -0.3167, "ip_prefix": "41.145"},
    "46 - Ain Temouchent": {"lat": 35.2975, "lon": -1.1404, "ip_prefix": "41.146"},
    "47 - Ghardaia": {"lat": 32.4909, "lon": 3.6735, "ip_prefix": "105.147"},
    "48 - Relizane": {"lat": 35.7373, "lon": 0.5560, "ip_prefix": "41.148"},
    "49 - Timimoun": {"lat": 29.2639, "lon": 0.2309, "ip_prefix": "105.149"},
    "50 - Bordj Badji Mokhtar": {"lat": 21.3289, "lon": 0.9542, "ip_prefix": "197.250"},
    "51 - Ouled Djellal": {"lat": 34.4167, "lon": 5.0667, "ip_prefix": "197.251"},
    "52 - Beni Abbes": {"lat": 30.1333, "lon": -2.1667, "ip_prefix": "41.152"},
    "53 - In Salah": {"lat": 27.1935, "lon": 2.4607, "ip_prefix": "197.253"},
    "54 - In Guezzam": {"lat": 19.5667, "lon": 5.7667, "ip_prefix": "197.254"},
    "55 - Touggourt": {"lat": 33.1053, "lon": 6.0580, "ip_prefix": "197.255"},
    "56 - Djanet": {"lat": 24.5542, "lon": 9.4844, "ip_prefix": "197.256"},
    "57 - El M'Ghair": {"lat": 33.9506, "lon": 5.9242, "ip_prefix": "197.257"},
    "58 - El Meniaa": {"lat": 30.5833, "lon": 2.8833, "ip_prefix": "105.158"},
}

foreign_proxies = {
    "Anonymizing VPN · Exit node: London": {"city": "London", "lat": 51.5074, "lon": -0.1278, "ip_prefix": "82.165"},
    "Anonymizing VPN · Exit node: Paris": {"city": "Paris", "lat": 48.8566, "lon": 2.3522, "ip_prefix": "193.56"},
    "Tor Exit Relay · Exit node: Frankfurt": {"city": "Frankfurt", "lat": 50.1109, "lon": 8.6821, "ip_prefix": "185.220"},
    "Anonymizing VPN · Exit node: Moscow": {"city": "Moscow", "lat": 55.7558, "lon": 37.6173, "ip_prefix": "95.173"},
    "SOCKS5 Relay · Exit node: Amsterdam": {"city": "Amsterdam", "lat": 52.3676, "lon": 4.9041, "ip_prefix": "194.109"},
    "Anonymizing VPN · Exit node: Madrid": {"city": "Madrid", "lat": 40.4168, "lon": -3.7038, "ip_prefix": "88.12"},
    "Tor Exit Relay · Exit node: Zurich": {"city": "Zurich", "lat": 47.3769, "lon": 8.5417, "ip_prefix": "178.209"},
}

products_catalog = [
    {"item": "ASUS ROG Strix Gaming Laptop", "category": "Electronics", "min": 240000, "max": 440000},
    {"item": "Samsung Galaxy S25 Ultra 512GB", "category": "Electronics", "min": 195000, "max": 310000},
    {"item": "Apple MacBook Pro 14 M3", "category": "Electronics", "min": 280000, "max": 490000},
    {"item": "iPhone 16 Pro Max 256GB", "category": "Electronics", "min": 215000, "max": 340000},
    {"item": "Sony PlayStation 5 Pro Bundle", "category": "Electronics", "min": 115000, "max": 165000},
    {"item": "NVIDIA RTX 4080 Super GPU", "category": "Electronics", "min": 185000, "max": 275000},
    {"item": "Apple iPad Pro 13-inch M4", "category": "Electronics", "min": 175000, "max": 265000},
    {"item": "DJI Mini 4 Pro Fly More Combo", "category": "Electronics", "min": 145000, "max": 210000},
    {"item": "LG C4 65-inch 4K OLED Smart TV", "category": "Home Appliances", "min": 210000, "max": 360000},
    {"item": "Samsung Bespoke Twin Refrigerator", "category": "Home Appliances", "min": 165000, "max": 280000},
    {"item": "Apple AirPods Pro 2 (USB-C)", "category": "Electronics", "min": 34000, "max": 49000},
    {"item": "Sony WH-1000XM5 Headphones", "category": "Electronics", "min": 46000, "max": 68000},
    {"item": "Logitech MX Master 3S Mouse", "category": "Electronics", "min": 16500, "max": 24000},
    {"item": "Keychron Q1 Mechanical Keyboard", "category": "Electronics", "min": 22000, "max": 35000},
    {"item": "Samsung Galaxy Watch 7 44mm", "category": "Electronics", "min": 38000, "max": 56000},
    {"item": "Xiaomi Redmi Note 14 Pro", "category": "Electronics", "min": 48000, "max": 72000},
    {"item": "Anker 737 PowerBank 24000mAh", "category": "Electronics", "min": 18000, "max": 29000},
    {"item": "De'Longhi Magnifica Espresso Machine", "category": "Home Appliances", "min": 55000, "max": 95000},
    {"item": "Dyson V15 Detect Cordless Vacuum", "category": "Home Appliances", "min": 85000, "max": 145000},
    {"item": "Philips XXL Digital Air Fryer", "category": "Home Appliances", "min": 24000, "max": 42000},
    {"item": "Brandt 12000 BTU Inverter AC", "category": "Home Appliances", "min": 68000, "max": 105000},
    {"item": "Condor 8kg Front Load Washer", "category": "Home Appliances", "min": 52000, "max": 78000},
    {"item": "Tefal Titanium Cookware Set (10pc)", "category": "Home Appliances", "min": 19000, "max": 34000},
    {"item": "Kenwood Titanium Stand Mixer", "category": "Home Appliances", "min": 36000, "max": 58000},
    {"item": "Nike Air Zoom Pegasus Running Shoes", "category": "Fashion & Clothing", "min": 14500, "max": 28000},
    {"item": "Adidas Predator Elite Football Boots", "category": "Fashion & Clothing", "min": 18000, "max": 34000},
    {"item": "North Face Insulated Winter Parka", "category": "Fashion & Clothing", "min": 22000, "max": 41000},
    {"item": "Artisanal Full-Grain Leather Briefcase", "category": "Fashion & Clothing", "min": 15000, "max": 32000},
    {"item": "Ray-Ban Polarized Wayfarer Sunglasses", "category": "Fashion & Clothing", "min": 19500, "max": 31000},
    {"item": "Casio G-Shock Tough Solar Watch", "category": "Fashion & Clothing", "min": 17000, "max": 29500},
    {"item": "Traditional Algerian Karakou Jacket", "category": "Fashion & Clothing", "min": 28000, "max": 65000},
    {"item": "Monthly Family Pantry Box (35 Items)", "category": "Groceries / Superette", "min": 8500, "max": 19500},
    {"item": "Kabylie Extra Virgin Olive Oil (5L)", "category": "Groceries / Superette", "min": 4800, "max": 8500},
    {"item": "Tolga Deglet Nour Dates (5kg Box)", "category": "Groceries / Superette", "min": 3200, "max": 6400},
    {"item": "Sidr Mountain Pure Honey (1kg)", "category": "Groceries / Superette", "min": 4500, "max": 9200},
    {"item": "Lavazza Crema Espresso Beans (2kg)", "category": "Groceries / Superette", "min": 5200, "max": 8800},
    {"item": "Optimum Gold Standard Whey 5lbs", "category": "Groceries / Superette", "min": 16000, "max": 24500},
    {"item": "PlayStation Plus 12-Month Pass", "category": "Digital Services", "min": 12000, "max": 22000},
    {"item": "Steam Wallet 100 USD Code", "category": "Digital Services", "min": 14000, "max": 24000},
    {"item": "Coursera Plus Annual Subscription", "category": "Digital Services", "min": 28000, "max": 52000},
]

FIRST_NAMES = [
    "Yassine", "Sarah", "Amine", "Karim", "Amina", "Mohamed", "Farid", "Nadia",
    "Sofiane", "Imane", "Mehdi", "Walid", "Samia", "Khaled", "Zohra", "Bilal",
    "Meriem", "Anis", "Selma", "Rachid", "Leila", "Hichem", "Khadidja", "Tarek",
    "Youcef", "Nesrine", "Abderrahmane", "Lina", "Hamza", "Rania", "Mourad", "Ryma",
    "Adel", "Chaima", "Lotfi", "Manel", "Nabil", "Ikram", "Sidali", "Yasmine"
]
LAST_NAMES = [
    "Benali", "Khelifi", "Mansouri", "Brahimi", "Saidi", "Boussaid", "Belkacem",
    "Cherif", "Zitouni", "Hamidi", "Bouazza", "Ghezzal", "Meziane", "Amrani",
    "Dahmani", "Haddad", "Slimani", "Touati", "Boudiaf", "Merbah", "Ouali",
    "Seddiki", "Guermah", "Kaci", "Bensalem", "Chaalal", "Derradji", "Ferhat",
    "Latreche", "Mebarki", "Nouri", "Rahmani", "Taleb", "Yahiaoui", "Zerrouki",
    "Ait-Ahmed", "Benmoussa", "Djebbar", "Kadri", "Mokhtari"
]

buyers_db = {}
all_operations = deque(maxlen=500)
audit_log = deque(maxlen=200)
pending_decline_queue = deque()  # Holds follow-up declined swipes so they stream sequentially!
sse_queues = set()
txn_counter = 10500
stream_interval_sec = 3.0
stream_paused = False


def log_audit(event: str, target: str, detail: str):
    audit_log.appendleft({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "event": event,
        "target": target,
        "detail": detail,
    })


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[0]}•••@{domain}" if local else "u•••@example.com"


def mask_phone(phone: str) -> str:
    digits = [c for c in phone if c.isdigit()]
    last2 = "".join(digits[-2:]) if len(digits) >= 2 else "42"
    prefix = digits[3] if len(digits) > 3 else "5"
    return f"+213 {prefix}•• •• •• {last2}"


def mask_ip(ip: str) -> str:
    parts = ip.split(".")
    return f"{parts[0]}.{parts[1]}.•••.•••" if len(parts) >= 2 else "105.101.•••.•••"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def get_or_create_buyer(username: Optional[str] = None):
    if username and username in buyers_db:
        return buyers_db[username]

    first = random.choice(FIRST_NAMES).strip()
    last = random.choice(LAST_NAMES).strip()
    full_name = f"{first} {last}"
    user_key = f"{first.lower()}_{last.lower().replace('-', '')}_{random.randint(10, 99)}"

    if user_key in buyers_db:
        return buyers_db[user_key]

    wilaya_name = random.choice(list(wilayas.keys()))
    coords = wilayas[wilaya_name]
    raw_email = f"{user_key}@example.com"
    raw_phone = f"+213 {random.choice(['550', '661', '770', '552', '662'])}-{random.randint(10,99)}-{random.randint(10,99)}-{random.randint(10,99)}"

    buyer = {
        "username": user_key,
        "full_name": full_name,
        "email": mask_email(raw_email),
        "email_masked": mask_email(raw_email),
        "phone": mask_phone(raw_phone),
        "phone_masked": mask_phone(raw_phone),
        "home_wilaya": wilaya_name,
        "is_banned": False,
        "ban_reason": None,
        "consecutive_declines": 0,
        "total_orders": 0,
        "total_spent_dzd": 0.0,
        "blocked_amount_dzd": 0.0,
        "home_lat": float(coords["lat"]),
        "home_lon": float(coords["lon"]),
        "last_verified_lat": float(coords["lat"]),
        "last_verified_lon": float(coords["lon"]),
        "last_verified_location": wilaya_name,
        "history": [],
    }
    buyers_db[user_key] = buyer
    return buyer


def update_buyer_Rollups(buyer: dict):
    hist = buyer["history"]
    buyer["total_orders"] = len(hist)
    buyer["total_spent_dzd"] = round(sum(h["amount"] for h in hist if h["action"] == "APPROVED"), 2)
    buyer["blocked_amount_dzd"] = round(sum(h["amount"] for h in hist if h["is_fraud"]), 2)


def generate_single_decline_step(username: str, step_num: int):
    """Generates step 1/3, 2/3 (warning/declined on dashboard), or 3/3 (auto-ban)."""
    global txn_counter
    t0 = time.perf_counter()
    txn_counter += 1
    now = datetime.now()
    buyer = get_or_create_buyer(username)

    loc_name = buyer["home_wilaya"]
    meta = wilayas[loc_name]
    lat = float(round(meta["lat"] + random.uniform(-0.012, 0.012), 4))
    lon = float(round(meta["lon"] + random.uniform(-0.012, 0.012), 4))
    raw_ip = f"{meta['ip_prefix']}.{random.randint(10, 250)}.{random.randint(10, 250)}"
    product = random.choice(products_catalog[:14])
    amount_dzd = float(round(random.uniform(product["min"], product["max"]), 2))

    buyer["consecutive_declines"] = step_num
    attempt = len(buyer["history"]) + 1
    home_coords = [float(round(buyer["home_lat"], 4)), float(round(buyer["home_lon"], 4))]

    if step_num < 3:
        # Pre-ban declined attempts visible live on the dashboard!
        is_fraud = False
        action = "PENDING_REVIEW" if step_num == 2 else "APPROVED"
        verdict = "DECLINED_WARNING"
        needs_review = (step_num == 2)
        fraud_prob = 0.42 if step_num == 1 else 0.68
        payment_status = f"Declined (Attempt {step_num}/3)"
        reasons = [f"Card declined ({step_num}/3)", "Card testing monitor"]
    else:
        # 3rd consecutive decline -> Automatic Ban!
        is_fraud = True
        action = "ORDER_BLOCKED_ACCOUNT_FROZEN"
        verdict = "FRAUD_BANNED"
        needs_review = False
        fraud_prob = round(random.uniform(0.91, 0.99), 4)
        payment_status = "Declined (Attempt 3/3)"
        reasons = ["Multiple declined attempts (3x)", "Card testing burst", "Account frozen"]
        buyer["is_banned"] = True
        buyer["ban_reason"] = f"Multiple Declined Attempts (3x consecutive card declines in {loc_name})"
        log_audit("AUTO_BAN", buyer["username"], buyer["ban_reason"])

    latency_ms = max(18, int((time.perf_counter() - t0) * 1000) + random.randint(18, 36))
    op_record = {
        "txn_id": f"ORD-DZ-{txn_counter}",
        "username": buyer["username"],
        "full_name": buyer["full_name"],
        "email": buyer["email_masked"],
        "email_masked": buyer["email_masked"],
        "phone_masked": buyer["phone_masked"],
        "ip_address": mask_ip(raw_ip),
        "ip_masked": mask_ip(raw_ip),
        "item": product["item"],
        "category": product["category"],
        "amount": amount_dzd,
        "amount_usd": round(amount_dzd / DZD_PER_USD, 2),
        "location": loc_name,
        "origin_city": loc_name,
        "prev_location": buyer["last_verified_location"],
        "coords": [lat, lon],
        "origin_coords": [lat, lon],
        "home_coords": home_coords,
        "prev_coords": home_coords,
        "attempt": attempt,
        "consecutive_declines": step_num,
        "distance_km": 2.4,
        "time_diff_min": 1.5,
        "velocity": 18.0,
        "payment_status": payment_status,
        "fraud_probability": fraud_prob,
        "impossible_travel": False,
        "multiple_declines": (step_num >= 3),
        "is_fraud": is_fraud,
        "verdict": verdict,
        "action": action,
        "needs_review": needs_review,
        "reasons": reasons,
        "buyer_is_banned": buyer["is_banned"],
        "ban_reason": buyer["ban_reason"],
        "latency_ms": latency_ms,
        "timestamp": now.strftime("%H:%M:%S"),
        "epoch": int(now.timestamp()),
    }
    buyer["history"].insert(0, op_record)
    update_buyer_Rollups(buyer)
    all_operations.append(op_record)
    return op_record


def generate_operation(force_mode: Optional[str] = None):
    global txn_counter
    # If there is a queued sequential card-decline step, emit it next!
    if force_mode is None and pending_decline_queue:
        u, step_n = pending_decline_queue.popleft()
        if not buyers_db[u]["is_banned"]:
            return generate_single_decline_step(u, step_n)

    t0 = time.perf_counter()
    txn_counter += 1
    now = datetime.now()

    banned_users = [u for u, b in buyers_db.items() if b["is_banned"]]
    active_users = [u for u, b in buyers_db.items() if not b["is_banned"]]

    if force_mode == "decline_sequence":
        target_buyer = random.choice(active_users) if active_users else get_or_create_buyer()["username"]
        pending_decline_queue.append((target_buyer, 2))
        pending_decline_queue.append((target_buyer, 3))
        return generate_single_decline_step(target_buyer, 1)

    if force_mode in ("legit", "attack", "review", "banned_retry"):
        scenario = force_mode
    else:
        r = random.random()
        if r < 0.14:
            scenario = "attack"
        elif r < 0.24:
            # Start a 3-step sequential card-decline burst!
            target_buyer = random.choice(active_users) if active_users else get_or_create_buyer()["username"]
            pending_decline_queue.append((target_buyer, 2))
            pending_decline_queue.append((target_buyer, 3))
            return generate_single_decline_step(target_buyer, 1)
        elif r < 0.32:
            scenario = "review"
        elif r < 0.38 and banned_users:
            scenario = "banned_retry"
        else:
            scenario = "legit"

    if scenario == "banned_retry" and banned_users:
        username = random.choice(banned_users)
        buyer = buyers_db[username]
    else:
        if not active_users or random.random() < 0.35:
            buyer = get_or_create_buyer()
            username = buyer["username"]
        else:
            username = random.choice(active_users)
            buyer = buyers_db[username]

    attempt = len(buyer["history"]) + 1
    home_coords = [float(round(buyer["home_lat"], 4)), float(round(buyer["home_lon"], 4))]
    anchor_lat = buyer["last_verified_lat"]
    anchor_lon = buyer["last_verified_lon"]
    prev_location = buyer["last_verified_location"]

    # CASE 1: BANNED ACCOUNT RETRY
    if buyer["is_banned"]:
        proxy_name = random.choice(list(foreign_proxies.keys()))
        p_meta = foreign_proxies[proxy_name]
        lat = float(round(p_meta["lat"] + random.uniform(-0.02, 0.02), 4))
        lon = float(round(p_meta["lon"] + random.uniform(-0.02, 0.02), 4))
        raw_ip = f"{p_meta['ip_prefix']}.{random.randint(10, 250)}.{random.randint(10, 250)}"
        product = random.choice(products_catalog[:10])
        amount_dzd = float(round(random.uniform(product["min"], product["max"]), 2))
        dist_km = round(haversine(anchor_lat, anchor_lon, lat, lon), 1)
        time_diff_min = round(random.uniform(5.0, 14.0), 1)
        velocity_kmh = round(dist_km / (time_diff_min / 60.0), 1)

        op_record = {
            "txn_id": f"ORD-DZ-{txn_counter}",
            "username": username,
            "full_name": buyer["full_name"],
            "email": buyer["email_masked"],
            "email_masked": buyer["email_masked"],
            "phone_masked": buyer["phone_masked"],
            "ip_address": mask_ip(raw_ip),
            "ip_masked": mask_ip(raw_ip),
            "item": product["item"],
            "category": product["category"],
            "amount": amount_dzd,
            "amount_usd": round(amount_dzd / DZD_PER_USD, 2),
            "location": proxy_name,
            "origin_city": p_meta["city"],
            "prev_location": prev_location,
            "coords": [lat, lon],
            "origin_coords": [lat, lon],
            "home_coords": home_coords,
            "prev_coords": home_coords,
            "attempt": attempt,
            "distance_km": dist_km,
            "time_diff_min": time_diff_min,
            "velocity": velocity_kmh,
            "payment_status": "Declined (Banned)",
            "fraud_probability": 0.99,
            "impossible_travel": velocity_kmh > 900.0,
            "is_fraud": True,
            "verdict": "BLOCKED (ACCOUNT BANNED)",
            "action": "REJECTED_BANNED_ACCOUNT",
            "needs_review": False,
            "reasons": ["Banned account reuse", "Gateway policy block"],
            "buyer_is_banned": True,
            "ban_reason": buyer["ban_reason"] or "Account Frozen",
            "latency_ms": random.randint(9, 20),
            "timestamp": now.strftime("%H:%M:%S"),
            "epoch": int(now.timestamp()),
        }
        buyer["history"].insert(0, op_record)
        update_buyer_Rollups(buyer)
        all_operations.append(op_record)
        return op_record

    # CASE 2: NEEDS HUMAN REVIEW
    if scenario == "review":
        other_wilaya = random.choice([w for w in wilayas.keys() if w != buyer["home_wilaya"]])
        meta = wilayas[other_wilaya]
        lat = float(round(meta["lat"] + random.uniform(-0.015, 0.015), 4))
        lon = float(round(meta["lon"] + random.uniform(-0.015, 0.015), 4))
        raw_ip = f"{meta['ip_prefix']}.{random.randint(10, 250)}.{random.randint(10, 250)}"
        product = random.choice(products_catalog[:12])
        amount_dzd = float(round(random.uniform(160000, 320000), 2))
        dist_km = round(haversine(anchor_lat, anchor_lon, lat, lon), 1)
        time_diff_min = round(random.uniform(35.0, 75.0), 1)
        velocity_kmh = round(dist_km / (time_diff_min / 60.0), 1)

        op_record = {
            "txn_id": f"ORD-DZ-{txn_counter}",
            "username": username,
            "full_name": buyer["full_name"],
            "email": buyer["email_masked"],
            "email_masked": buyer["email_masked"],
            "phone_masked": buyer["phone_masked"],
            "ip_address": mask_ip(raw_ip),
            "ip_masked": mask_ip(raw_ip),
            "item": product["item"],
            "category": product["category"],
            "amount": amount_dzd,
            "amount_usd": round(amount_dzd / DZD_PER_USD, 2),
            "location": other_wilaya,
            "origin_city": other_wilaya,
            "prev_location": prev_location,
            "coords": [lat, lon],
            "origin_coords": [lat, lon],
            "home_coords": home_coords,
            "prev_coords": home_coords,
            "attempt": attempt,
            "distance_km": dist_km,
            "time_diff_min": time_diff_min,
            "velocity": velocity_kmh,
            "payment_status": "On Hold (Review)",
            "fraud_probability": round(random.uniform(0.45, 0.68), 4),
            "impossible_travel": False,
            "is_fraud": False,
            "verdict": "PENDING_REVIEW",
            "action": "PENDING_REVIEW",
            "needs_review": True,
            "reasons": ["Unusual Wilaya", "High-value order"],
            "buyer_is_banned": False,
            "ban_reason": None,
            "latency_ms": random.randint(24, 42),
            "timestamp": now.strftime("%H:%M:%S"),
            "epoch": int(now.timestamp()),
        }
        buyer["history"].insert(0, op_record)
        update_buyer_Rollups(buyer)
        all_operations.append(op_record)
        return op_record

    # CASE 3: IMPOSSIBLE TRAVEL ATTACK OR LEGITIMATE ORDER
    if scenario == "attack":
        loc_name = random.choice(list(foreign_proxies.keys()))
        meta = foreign_proxies[loc_name]
        origin_city = meta["city"]
        lat = float(round(meta["lat"] + random.uniform(-0.02, 0.02), 4))
        lon = float(round(meta["lon"] + random.uniform(-0.02, 0.02), 4))
        raw_ip = f"{meta['ip_prefix']}.{random.randint(10, 250)}.{random.randint(10, 250)}"
        product = random.choice(products_catalog[:10])
        amount_dzd = float(round(random.uniform(product["min"], product["max"]), 2))
        card_declined = random.random() < 0.55
        time_diff_min = float(round(random.uniform(6.0, 14.0), 1))
        dist_km = float(round(haversine(anchor_lat, anchor_lon, lat, lon), 1))
        velocity_kmh = float(round(dist_km / (time_diff_min / 60.0), 1))
    else:
        loc_name = buyer["home_wilaya"]
        meta = wilayas[loc_name]
        origin_city = loc_name
        lat = float(round(meta["lat"] + random.uniform(-0.015, 0.015), 4))
        lon = float(round(meta["lon"] + random.uniform(-0.015, 0.015), 4))
        raw_ip = f"{meta['ip_prefix']}.{random.randint(10, 250)}.{random.randint(10, 250)}"
        product = random.choice(products_catalog[10:])
        amount_dzd = float(round(random.uniform(product["min"], product["max"]), 2))
        card_declined = False
        dist_km = 0.0 if attempt == 1 else float(round(haversine(anchor_lat, anchor_lon, lat, lon), 1))
        time_diff_min = 0.0 if attempt == 1 else float(round(random.uniform(65.0, 180.0), 1))
        velocity_kmh = 0.0 if attempt == 1 else float(round(dist_km / (time_diff_min / 60.0), 1))
        buyer["last_verified_lat"] = lat
        buyer["last_verified_lon"] = lon
        buyer["last_verified_location"] = loc_name

    feat_dict = {
        "Amount (DZD)": amount_dzd,
        "Velocity (KM/H)": velocity_kmh,
        "Attempt Number": attempt,
        "is_declined": 1 if card_declined else 0,
        "Latitude": lat,
        "Longitude": lon,
    }
    for col in feature_cols:
        if col.startswith("Merchant Category_"):
            feat_dict[col] = 1.0 if col == f"Merchant Category_{product['category']}" else 0.0

    vec = np.array([[feat_dict.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
    vec_scaled = scaler.transform(vec)

    fraud_prob = float(model(vec_scaled, training=False)[0][0])
    impossible_travel = bool(velocity_kmh > 900.0)
    is_fraud_detected = bool(impossible_travel or (fraud_prob >= 0.50))

    if is_fraud_detected and fraud_prob < 0.88:
        fraud_prob = round(random.uniform(0.89, 0.994), 4)

    reasons = []
    if impossible_travel:
        reasons.append("Impossible travel")
    if "Exit" in loc_name or "Relay" in loc_name:
        reasons.append("Foreign VPN / Tor")
    if amount_dzd >= 180000:
        reasons.append("High-value order")
    if card_declined:
        reasons.append("Card declined")
    if not reasons:
        reasons.append("Verified home Wilaya")

    if is_fraud_detected:
        verdict = "FRAUD_BANNED"
        action = "ORDER_BLOCKED_ACCOUNT_FROZEN"
        payment_status = "Declined (Blocked)" if card_declined else "Blocked at Gateway"
        buyer["is_banned"] = True
        buyer["ban_reason"] = f"Impossible travel ({velocity_kmh:,.0f} km/h via {origin_city})"
        log_audit("AUTO_BAN", username, buyer["ban_reason"])
    else:
        verdict = "APPROVED"
        action = "APPROVED"
        payment_status = "Authorized"

    latency_ms = max(18, int((time.perf_counter() - t0) * 1000) + random.randint(20, 42))

    op_record = {
        "txn_id": f"ORD-DZ-{txn_counter}",
        "username": username,
        "full_name": buyer["full_name"],
        "email": buyer["email_masked"],
        "email_masked": buyer["email_masked"],
        "phone_masked": buyer["phone_masked"],
        "ip_address": mask_ip(raw_ip),
        "ip_masked": mask_ip(raw_ip),
        "item": product["item"],
        "category": product["category"],
        "amount": amount_dzd,
        "amount_usd": round(amount_dzd / DZD_PER_USD, 2),
        "location": loc_name,
        "origin_city": origin_city,
        "prev_location": prev_location,
        "coords": [lat, lon],
        "origin_coords": [lat, lon],
        "home_coords": home_coords,
        "prev_coords": [float(round(anchor_lat, 4)), float(round(anchor_lon, 4))],
        "attempt": attempt,
        "distance_km": dist_km,
        "time_diff_min": time_diff_min,
        "velocity": velocity_kmh,
        "payment_status": payment_status,
        "fraud_probability": round(fraud_prob, 4),
        "impossible_travel": impossible_travel,
        "is_fraud": is_fraud_detected,
        "verdict": verdict,
        "action": action,
        "needs_review": False,
        "reasons": reasons,
        "buyer_is_banned": buyer["is_banned"],
        "ban_reason": buyer["ban_reason"],
        "latency_ms": latency_ms,
        "timestamp": now.strftime("%H:%M:%S"),
        "epoch": int(now.timestamp()),
    }
    buyer["history"].insert(0, op_record)
    update_buyer_Rollups(buyer)
    all_operations.append(op_record)
    return op_record


def broadcast_sse(event_name: str, payload: dict):
    for q in list(sse_queues):
        try:
            q.put_nowait((event_name, payload))
        except Exception:
            pass


def init_seed_data():
    global buyers_db, all_operations, txn_counter
    buyers_db.clear()
    all_operations.clear()
    audit_log.clear()
    pending_decline_queue.clear()
    txn_counter = 10500
    for _ in range(35):
        get_or_create_buyer()
    for mode in ["legit", "legit", "attack", "review", "decline_sequence"]:
        generate_operation(force_mode=mode)
    log_audit("SYSTEM_BOOT", "SentinelAI", "Initialized 58 Wilayas monitoring & neural model")


init_seed_data()


@app.on_event("startup")
async def start_background_stream():
    async def loop():
        while True:
            await asyncio.sleep(stream_interval_sec)
            if not stream_paused and sse_queues:
                op = await asyncio.to_thread(generate_operation)
                broadcast_sse("order", op)

    asyncio.create_task(loop())


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/audit")
def get_audit_log():
    return list(audit_log)


@app.get("/api/stats")
def get_stats():
    ops = list(all_operations)
    total = len(ops)
    approved_vol = sum(o["amount"] for o in ops if o["action"] == "APPROVED")
    fraud_prev = sum(o["amount"] for o in ops if o["is_fraud"])
    held_vol = sum(o["amount"] for o in ops if o["action"] == "PENDING_REVIEW")
    attacks_blocked = sum(1 for o in ops if o["is_fraud"])
    banned_count = sum(1 for b in buyers_db.values() if b["is_banned"])
    avg_lat = int(np.mean([o["latency_ms"] for o in ops])) if ops else 38

    cat_counts = {}
    origin_counts = {}
    for o in ops:
        if o["is_fraud"]:
            cat = o["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            orig = o.get("origin_city") or o["location"]
            origin_counts[orig] = origin_counts.get(orig, 0) + 1

    top_origins = [
        {"name": k, "count": v}
        for k, v in sorted(origin_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    ]

    now_str = datetime.now().strftime("%H:%M")
    orders_per_min = []
    chunk_size = max(1, len(ops) // 12)
    for i in range(12):
        slice_ops = ops[i * chunk_size : (i + 1) * chunk_size]
        orders_per_min.append({
            "t": f"-{12 - i}m" if i < 11 else now_str,
            "approved": sum(1 for x in slice_ops if x["action"] == "APPROVED"),
            "review": sum(1 for x in slice_ops if x["action"] == "PENDING_REVIEW"),
            "blocked": sum(1 for x in slice_ops if x["is_fraud"]),
        })

    return {
        "total_orders": total,
        "approved_volume_dzd": round(approved_vol, 2),
        "approved_volume_usd": round(approved_vol / DZD_PER_USD, 2),
        "fraud_prevented_dzd": round(fraud_prev, 2),
        "fraud_prevented_usd": round(fraud_prev / DZD_PER_USD, 2),
        "held_amount_dzd": round(held_vol, 2),
        "attacks_blocked": attacks_blocked,
        "banned_accounts": banned_count,
        "avg_latency_ms": avg_lat,
        "wilayas_monitored": len(wilayas),
        "orders_per_minute": orders_per_min,
        "blocked_by_category": cat_counts,
        "top_origins": top_origins,
    }


@app.get("/api/stream")
async def sse_stream():
    q = asyncio.Queue(maxsize=100)
    sse_queues.add(q)

    async def event_generator():
        try:
            while True:
                ev_name, data = await q.get()
                yield f"event: {ev_name}\ndata: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            sse_queues.discard(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/next-order")
def get_next_order():
    op = generate_operation()
    broadcast_sse("order", op)
    return op


@app.get("/api/buyers")
def get_all_buyers(banned_only: bool = False, include_history: bool = True):
    res = []
    for b in buyers_db.values():
        if banned_only and not b["is_banned"]:
            continue
        if include_history:
            res.append(b)
        else:
            copy_b = {k: v for k, v in b.items() if k != "history"}
            res.append(copy_b)
    return res


@app.get("/api/buyer/{username}")
def get_buyer_details(username: str):
    if username not in buyers_db:
        raise HTTPException(status_code=404, detail="Buyer not found")
    return buyers_db[username]


@app.post("/api/review/{txn_id}/{decision}")
def review_order(txn_id: str, decision: str):
    target_op = next((o for o in all_operations if o["txn_id"] == txn_id), None)
    if not target_op:
        raise HTTPException(status_code=404, detail="Order not found")

    buyer = buyers_db.get(target_op["username"])
    if decision == "approve":
        target_op["action"] = "APPROVED"
        target_op["verdict"] = "APPROVED"
        target_op["needs_review"] = False
        target_op["is_fraud"] = False
        target_op["payment_status"] = "Authorized (Analyst)"
        log_audit("ANALYST_APPROVE", txn_id, f"Approved order for {target_op['full_name']}")
    elif decision == "block":
        target_op["action"] = "ORDER_BLOCKED_ACCOUNT_FROZEN"
        target_op["verdict"] = "FRAUD_BANNED"
        target_op["needs_review"] = False
        target_op["is_fraud"] = True
        target_op["payment_status"] = "Blocked (Analyst)"
        if buyer:
            buyer["is_banned"] = True
            buyer["ban_reason"] = f"Analyst Blocked ({txn_id})"
            target_op["buyer_is_banned"] = True
        log_audit("ANALYST_BLOCK", txn_id, f"Blocked & froze {target_op['full_name']}")
    else:
        raise HTTPException(status_code=400, detail="Invalid decision")

    if buyer:
        update_buyer_Rollups(buyer)
    broadcast_sse("review", target_op)
    return target_op


@app.post("/api/unban/{username}")
def unban_buyer(username: str):
    if username not in buyers_db:
        raise HTTPException(status_code=404, detail="Buyer not found")
    buyer = buyers_db[username]
    buyer["is_banned"] = False
    buyer["ban_reason"] = None
    buyer["consecutive_declines"] = 0
    buyer["last_verified_lat"] = buyer["home_lat"]
    buyer["last_verified_lon"] = buyer["home_lon"]
    buyer["last_verified_location"] = buyer["home_wilaya"]
    for h in buyer["history"]:
        h["buyer_is_banned"] = False
    log_audit("ANALYST_UNBAN", username, f"Unbanned {buyer['full_name']} & reset home Wilaya")
    broadcast_sse("unban", {"username": username})
    return {"status": "success", "username": username, "buyer": buyer}


@app.post("/api/demo/speed")
def set_demo_speed(seconds: float = Query(3.0, ge=0.5, le=30.0)):
    global stream_interval_sec
    stream_interval_sec = float(seconds)
    return {"status": "ok", "stream_interval_sec": stream_interval_sec}


@app.post("/api/demo/{action_name}")
def trigger_demo_action(action_name: str):
    global stream_paused
    if action_name == "pause":
        stream_paused = True
        return {"status": "ok", "paused": True}
    if action_name == "resume":
        stream_paused = False
        return {"status": "ok", "paused": False}
    if action_name == "reset":
        init_seed_data()
        broadcast_sse("reset", {"status": "reset"})
        return {"status": "reset"}
    if action_name == "burst":
        burst_ops = []
        for _ in range(10):
            op = generate_operation()
            broadcast_sse("order", op)
            burst_ops.append(op)
        return {"status": "ok", "operations": burst_ops}
    if action_name in ("legit", "attack", "decline_sequence", "review", "banned_retry"):
        op = generate_operation(force_mode=action_name)
        broadcast_sse("order", op)
        return op
    raise HTTPException(status_code=400, detail="Unknown demo action")


@app.get("/api/export.csv")
def export_csv(username: Optional[str] = None):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Order ID", "Timestamp", "Buyer Name", "Username", "Masked Email",
        "Product", "Category", "Amount (DZD)", "Amount (USD)", "Prev Location",
        "Origin City", "Distance (km)", "Time Diff (min)", "Velocity (km/h)",
        "Payment Status", "Risk (%)", "Action", "Reasons"
    ])
    ops = buyers_db[username]["history"] if (username and username in buyers_db) else list(all_operations)
    for o in ops:
        writer.writerow([
            o["txn_id"], o["timestamp"], o["full_name"], o["username"],
            o.get("email_masked", o["email"]), o["item"], o["category"],
            o["amount"], o.get("amount_usd", ""), o.get("prev_location", ""),
            o.get("origin_city", o["location"]), o.get("distance_km", 0), o.get("time_diff_min", 0),
            o["velocity"], o.get("payment_status", ""), round(o["fraud_probability"] * 100, 1),
            o.get("action", o["verdict"]), "; ".join(o.get("reasons", []))
        ])
    output.seek(0)
    filename = f"sentinel_case_{username}.csv" if username else "sentinel_incident_report.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )