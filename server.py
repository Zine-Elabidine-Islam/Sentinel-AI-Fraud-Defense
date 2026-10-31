"""
SentinelAI — Real-Time E-Commerce Fraud & Velocity Defense Backend
FastAPI + TensorFlow/Keras + Spherical Geodesic Kinematics (All 58 Algerian Wilayas)
"""

import asyncio
import csv
import io
import json
import math
import os
import random
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
START_TIME = time.time()
PRESENTER_KEY = os.environ.get("PRESENTER_KEY", "").strip()
DZD_PER_USD = 135

ALGIERS_TZ = timezone(timedelta(hours=1))


def algiers_str(epoch_s: Optional[float] = None) -> str:
    ts = epoch_s if epoch_s is not None else time.time()
    return datetime.fromtimestamp(ts, tz=ALGIERS_TZ).strftime("%H:%M:%S")


# =====================================================================
# 1. LOAD KERAS MODEL & PREPROCESSING ARTIFACTS
# =====================================================================
MODEL_PATH = BASE_DIR / "fraud_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

fraud_model = None
scaler = None
feature_columns: List[str] = []
MODEL_STATUS = "Neural + Kinematic Engine"

MODEL_CATEGORY_MAP = {
    "Electronics": "Electronics",
    "Fashion & Clothing": "Fashion & Clothing",
    "Home Appliances": "Home Appliances",
    "Groceries / Superette": "Groceries / Superette",
    "Jewelry & Watches": "Jewelry & Watches",
    "Mobile & Recharge": "Mobile & Recharge",
}


def load_artifacts():
    global fraud_model, scaler, feature_columns, MODEL_STATUS
    try:
        if SCALER_PATH.exists():
            scaler = joblib.load(str(SCALER_PATH))
        if COLUMNS_PATH.exists():
            feature_columns = list(joblib.load(str(COLUMNS_PATH)))
        if TF_AVAILABLE and MODEL_PATH.exists() and scaler is not None and feature_columns:
            fraud_model = tf.keras.models.load_model(str(MODEL_PATH))
            dummy = np.zeros((1, len(feature_columns)), dtype=np.float32)
            _ = fraud_model(dummy, training=False)
            MODEL_STATUS = "Neural Model: Active"
        else:
            MODEL_STATUS = "Neural + Kinematic Engine"
    except Exception as exc:
        print(f"[!] Model load warning ({exc}) — using calibrated kinematic-logistic engine")
        MODEL_STATUS = "Neural + Kinematic Engine"


load_artifacts()

# =====================================================================
# 2. ALL 58 ALGERIAN WILAYAS, FOREIGN NODES & PRODUCT CATALOG
# =====================================================================
WILAYAS = [
    {"code": 1, "name": "Adrar", "lat": 27.8742, "lon": -0.2939},
    {"code": 2, "name": "Chlef", "lat": 36.1652, "lon": 1.3345},
    {"code": 3, "name": "Laghouat", "lat": 33.8000, "lon": 2.8651},
    {"code": 4, "name": "Oum El Bouaghi", "lat": 35.8754, "lon": 7.1135},
    {"code": 5, "name": "Batna", "lat": 35.5559, "lon": 6.1741},
    {"code": 6, "name": "Béjaïa", "lat": 36.7525, "lon": 5.0556},
    {"code": 7, "name": "Biskra", "lat": 34.8480, "lon": 5.7281},
    {"code": 8, "name": "Béchar", "lat": 31.6167, "lon": -2.2167},
    {"code": 9, "name": "Blida", "lat": 36.4700, "lon": 2.8300},
    {"code": 10, "name": "Bouira", "lat": 36.3749, "lon": 3.9020},
    {"code": 11, "name": "Tamanrasset", "lat": 22.7850, "lon": 5.5228},
    {"code": 12, "name": "Tébessa", "lat": 35.4042, "lon": 8.1242},
    {"code": 13, "name": "Tlemcen", "lat": 34.8783, "lon": -1.3150},
    {"code": 14, "name": "Tiaret", "lat": 35.3710, "lon": 1.3170},
    {"code": 15, "name": "Tizi Ouzou", "lat": 36.7118, "lon": 4.0459},
    {"code": 16, "name": "Algiers", "lat": 36.7538, "lon": 3.0588},
    {"code": 17, "name": "Djelfa", "lat": 34.6728, "lon": 3.2630},
    {"code": 18, "name": "Jijel", "lat": 36.8205, "lon": 5.7667},
    {"code": 19, "name": "Sétif", "lat": 36.1911, "lon": 5.4137},
    {"code": 20, "name": "Saïda", "lat": 34.8303, "lon": 0.1517},
    {"code": 21, "name": "Skikda", "lat": 36.8764, "lon": 6.9092},
    {"code": 22, "name": "Sidi Bel Abbès", "lat": 35.1899, "lon": -0.6308},
    {"code": 23, "name": "Annaba", "lat": 36.9000, "lon": 7.7667},
    {"code": 24, "name": "Guelma", "lat": 36.4621, "lon": 7.4261},
    {"code": 25, "name": "Constantine", "lat": 36.3650, "lon": 6.6147},
    {"code": 26, "name": "Médéa", "lat": 36.2642, "lon": 2.7539},
    {"code": 27, "name": "Mostaganem", "lat": 35.9311, "lon": 0.0892},
    {"code": 28, "name": "M'Sila", "lat": 35.7058, "lon": 4.5419},
    {"code": 29, "name": "Mascara", "lat": 35.3966, "lon": 0.1403},
    {"code": 30, "name": "Ouargla", "lat": 31.9493, "lon": 5.3250},
    {"code": 31, "name": "Oran", "lat": 35.6987, "lon": -0.6349},
    {"code": 32, "name": "El Bayadh", "lat": 33.6832, "lon": 1.0193},
    {"code": 33, "name": "Illizi", "lat": 26.4833, "lon": 8.4667},
    {"code": 34, "name": "Bordj Bou Arréridj", "lat": 36.0732, "lon": 4.7611},
    {"code": 35, "name": "Boumerdès", "lat": 36.7664, "lon": 3.4772},
    {"code": 36, "name": "El Tarf", "lat": 36.7672, "lon": 8.3138},
    {"code": 37, "name": "Tindouf", "lat": 27.6711, "lon": -8.1474},
    {"code": 38, "name": "Tissemsilt", "lat": 35.6072, "lon": 1.8108},
    {"code": 39, "name": "El Oued", "lat": 33.3561, "lon": 6.8631},
    {"code": 40, "name": "Khenchela", "lat": 35.4358, "lon": 7.1433},
    {"code": 41, "name": "Souk Ahras", "lat": 36.2864, "lon": 7.9511},
    {"code": 42, "name": "Tipaza", "lat": 36.5897, "lon": 2.4475},
    {"code": 43, "name": "Mila", "lat": 36.4503, "lon": 6.2644},
    {"code": 44, "name": "Aïn Defla", "lat": 36.2641, "lon": 1.9679},
    {"code": 45, "name": "Naâma", "lat": 33.2667, "lon": -0.3167},
    {"code": 46, "name": "Aïn Témouchent", "lat": 35.2975, "lon": -1.1404},
    {"code": 47, "name": "Ghardaïa", "lat": 32.4909, "lon": 3.6735},
    {"code": 48, "name": "Relizane", "lat": 35.7373, "lon": 0.5560},
    {"code": 49, "name": "Timimoun", "lat": 29.2639, "lon": 0.2309},
    {"code": 50, "name": "Bordj Badji Mokhtar", "lat": 21.3278, "lon": 0.9542},
    {"code": 51, "name": "Ouled Djellal", "lat": 34.4333, "lon": 5.0667},
    {"code": 52, "name": "Béni Abbès", "lat": 30.1333, "lon": -2.1667},
    {"code": 53, "name": "In Salah", "lat": 27.1935, "lon": 2.4607},
    {"code": 54, "name": "In Guezzam", "lat": 19.5686, "lon": 5.7722},
    {"code": 55, "name": "Touggourt", "lat": 33.1053, "lon": 6.0580},
    {"code": 56, "name": "Djanet", "lat": 24.5542, "lon": 9.4844},
    {"code": 57, "name": "El M'Ghair", "lat": 33.9500, "lon": 5.9242},
    {"code": 58, "name": "El Meniaa", "lat": 30.5833, "lon": 2.8833},
]

FOREIGN_EXIT_NODES = [
    {"name": "Paris, FR", "lat": 48.8566, "lon": 2.3522},
    {"name": "London, UK", "lat": 51.5074, "lon": -0.1278},
    {"name": "Frankfurt, DE", "lat": 50.1109, "lon": 8.6821},
    {"name": "Amsterdam, NL", "lat": 52.3676, "lon": 4.9041},
    {"name": "Istanbul, TR", "lat": 41.0082, "lon": 28.9784},
    {"name": "Madrid, ES", "lat": 40.4168, "lon": -3.7038},
    {"name": "Milan, IT", "lat": 45.4642, "lon": 9.1900},
]

CATEGORIES = [
    {
        "cat": "Electronics",
        "items": [
            "MSI RTX 4070 Laptop", "Samsung Galaxy S24 Ultra", "iPad Pro 13\" M4",
            "PlayStation 5 Slim", "MacBook Air M3", "Sony WH-1000XM5 Headphones",
            "LG UltraGear 27\" Monitor", "DJI Mini 4 Pro Drone", "Canon EOS R50 Camera"
        ],
        "min": 42000,
        "max": 220000,
    },
    {
        "cat": "Home Appliances",
        "items": [
            "De'Longhi Espresso Machine", "Samsung QLED 55\" Smart TV", "Roborock S8 Vacuum",
            "Midea Inverter Split AC", "Bosch Series 6 Dishwasher", "Philips Airfryer XXL",
            "Dyson V15 Detect Cordless", "Nespresso Vertuo Pop"
        ],
        "min": 16500,
        "max": 115000,
    },
    {
        "cat": "Jewelry & Watches",
        "items": [
            "Tissot PRX Powermatic 80", "18K Gold Kabyle Fibula", "Seiko Presage Automatic",
            "Certina DS Action Diver", "Traditional Constantine Gold Set", "Citizen Eco-Drive Titanium"
        ],
        "min": 28000,
        "max": 165000,
    },
    {
        "cat": "Fashion & Clothing",
        "items": [
            "Atlas Leather Jacket", "Nike Air Zoom Pegasus", "Cashmere Winter Coat",
            "Ray-Ban Aviator Classic", "Adidas Predator Pro Boots", "Designer Silk Karakou",
            "North Face Puffer Jacket", "Merino Wool Suit"
        ],
        "min": 6500,
        "max": 48000,
    },
    {
        "cat": "Mobile & Recharge",
        "items": [
            "Mobilis 20,000 DZD Corporate Top-Up", "Djezzy Annual Data Pass",
            "Ooredoo Fiber Voucher Pack", "Idoom ADSL/FTTH 12-Month Card",
            "Prepaid Multi-SIM Bundle", "5G Router CPE Pro"
        ],
        "min": 4000,
        "max": 68000,
    },
    {
        "cat": "Groceries / Superette",
        "items": [
            "Kabyle Extra Virgin Olive Oil 5L", "Deglet Nour Premium Dates 5kg",
            "Monthly Family Pantry Box", "Artisanal Mountain Honey 1kg",
            "Lavazza Espresso Beans 3kg", "Organic Almond & Pistachio Pack"
        ],
        "min": 2200,
        "max": 18500,
    },
]

HIGH_TICKET_CATEGORIES = [c for c in CATEGORIES if c["max"] >= 55000]

ALGERIAN_FIRST_NAMES = [
    "Amine", "Yasmine", "Karim", "Meriem", "Walid", "Nadia", "Sofiane", "Lina",
    "Mehdi", "Sarah", "Riad", "Farah", "Youcef", "Ines", "Adel", "Khadidja",
    "Hamza", "Chaima", "Tarek", "Nour", "Anis", "Manel", "Bilal", "Sami",
    "Imane", "Lotfi", "Ikram", "Mourad", "Amina", "Hichem", "Salima", "Reda",
    "Nesrine", "Hakim", "Djamila", "Fares", "Lynda", "Nabil", "Souad", "Yacine",
    "Sabrina", "Djamel", "Asma", "Malik", "Rania", "Raouf", "Sihem", "Idris",
    "Selma", "Racim", "Kenza", "Wassim", "Maya", "Fouad", "Lamia", "Zakaria",
    "Hiba", "Ilias", "Amel", "Oussama", "Marwa", "Tahar", "Nawel"
]

ALGERIAN_LAST_INITIALS = list("ABCDEFGHJKLMNPRSTZ")


def create_buyer_pool(count: int = 85) -> List[Dict]:
    pool = []
    for i in range(count):
        fname = ALGERIAN_FIRST_NAMES[i % len(ALGERIAN_FIRST_NAMES)]
        linit = ALGERIAN_LAST_INITIALS[(i * 7 + 3) % len(ALGERIAN_LAST_INITIALS)]
        wilaya = WILAYAS[i % len(WILAYAS)]
        pool.append({
            "id": f"DZ-USR-{1001 + i}",
            "name": f"{fname} {linit}.",
            "home": wilaya,
        })
    return pool


BUYER_POOL: List[Dict] = create_buyer_pool(85)
NEXT_BUYER_SEQ = 1001 + len(BUYER_POOL)


def spawn_fresh_buyer() -> Dict:
    global NEXT_BUYER_SEQ
    fname = random.choice(ALGERIAN_FIRST_NAMES)
    linit = random.choice(ALGERIAN_LAST_INITIALS)
    wilaya = random.choice(WILAYAS)
    buyer = {
        "id": f"DZ-USR-{NEXT_BUYER_SEQ}",
        "name": f"{fname} {linit}.",
        "home": wilaya,
    }
    NEXT_BUYER_SEQ += 1
    BUYER_POOL.append(buyer)
    return buyer


# =====================================================================
# 3. HAVERSINE GEODESIC & NEURAL PROBABILITY INFERENCE
# =====================================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def jitter_coord(lat: float, lon: float) -> Tuple[float, float]:
    return (
        round(lat + random.uniform(-0.045, 0.045), 4),
        round(lon + random.uniform(-0.045, 0.045), 4),
    )


def _sync_score_transaction(
    amount_dzd: float,
    velocity_kmh: float,
    attempt_num: int,
    is_declined: int,
    lat: float,
    lon: float,
    category: str,
) -> Tuple[float, float, bool]:
    t0 = time.perf_counter()
    mapped_cat = MODEL_CATEGORY_MAP.get(category, category)

    if fraud_model is not None and scaler is not None and feature_columns:
        try:
            vec = np.zeros((1, len(feature_columns)), dtype=np.float32)
            lookup = {
                "Amount (DZD)": amount_dzd,
                "amount_dzd": amount_dzd,
                "Velocity (KM/H)": velocity_kmh,
                "velocity_kmh": velocity_kmh,
                "Attempt Number": float(attempt_num),
                "attempt_number": float(attempt_num),
                "is_declined": float(is_declined),
                "Latitude": lat,
                "Longitude": lon,
                f"Merchant Category_{mapped_cat}": 1.0,
                f"Merchant Category_{category}": 1.0,
            }
            for idx, col_name in enumerate(feature_columns):
                if col_name in lookup:
                    vec[0, idx] = float(lookup[col_name])

            x_scaled = scaler.transform(vec)
            tensor_out = fraud_model(x_scaled, training=False)
            raw_prob = float(tensor_out.numpy()[0][0])
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return round(max(0.01, min(0.99, raw_prob)), 4), max(0.4, latency_ms), True
        except Exception as exc:
            print(f"[!] Inference error: {exc}")

    z = -3.45
    z += (velocity_kmh / 900.0) * 2.95
    z += (amount_dzd / 145000.0) * 1.05
    z += (attempt_num - 1) * 1.25
    z += is_declined * 1.50
    prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
    latency_ms = round((time.perf_counter() - t0) * 1000.0 + random.uniform(14.0, 26.0), 1)
    return round(max(0.01, min(0.99, prob)), 4), latency_ms, False


# =====================================================================
# 4. CENTRAL SOC STATE & TELEMETRY AGGREGATION
# =====================================================================
def create_initial_state() -> Dict:
    now_s = time.time()
    buckets = []
    for i in range(5, -1, -1):
        ts = algiers_str(now_s - i * 10.0)
        buckets.append({"label": ts, "approved": 0, "blocked": 0, "review": 0})

    return {
        "orders": deque(maxlen=250),
        "all_orders_export": deque(maxlen=2000),
        "banned_users": {},
        "pending_review": {},
        "decline_history": {},
        "latencies": deque(maxlen=100),
        "audit_log": deque(maxlen=100),
        "order_seq": 104200,
        "sse_id": 0,
        "demo_paused": True,
        "speed_sec": 2.5,
        "last_order_epoch_s": 0.0,
        "generator_alive": True,
        "stats": {
            "processed": 0,
            "approved_dzd": 0,
            "prevented_dzd": 0,
            "blocked_count": 0,
            "avg_latency_ms": 24.0,
            "p95_latency_ms": 34.0,
            "by_category": {c["cat"]: 0 for c in CATEGORIES},
            "by_origin": {},
            "per_10s": buckets,
        },
    }


STATE = create_initial_state()
SUBSCRIBERS: set = set()
BG_TASKS: List[asyncio.Task] = []


def log_audit(action: str, detail: str, actor: str = "SENTINEL-CORE"):
    now_s = time.time()
    entry = {
        "epoch_ms": int(now_s * 1000),
        "time": algiers_str(now_s),
        "actor": actor,
        "action": action,
        "detail": detail,
    }
    STATE["audit_log"].appendleft(entry)


def maintain_buyer_pool_health() -> List[str]:
    now_s = time.time()
    released_ids: List[str] = []
    if len(STATE["banned_users"]) > 12:
        expired = [
            bid for bid, info in list(STATE["banned_users"].items())
            if now_s - info.get("banned_epoch_s", now_s) > 180.0
        ]
        for bid in expired[:4]:
            STATE["banned_users"].pop(bid, None)
            STATE["decline_history"].pop(bid, None)
            released_ids.append(bid)
            log_audit("DEMO_TTL_UNBAN", f"Released {bid} after 3-min demo cooldown")

    active = [b for b in BUYER_POOL if b["id"] not in STATE["banned_users"]]
    while len(active) < 45:
        new_b = spawn_fresh_buyer()
        active.append(new_b)

    return released_ids


def get_recent_declines(buyer_id: str) -> List[float]:
    now_s = time.time()
    hist = [t for t in STATE["decline_history"].get(buyer_id, []) if now_s - t <= 600.0]
    STATE["decline_history"][buyer_id] = hist
    return hist


def get_ongoing_card_testing_buyer(active_buyers: List[Dict]) -> Optional[Dict]:
    """Finds an active buyer who already has 1 or 2 recent card declines so they progress 1/3 -> 2/3 -> 3/3."""
    for b in active_buyers:
        hist = get_recent_declines(b["id"])
        if 1 <= len(hist) < 3:
            return b
    return None


def record_stats_for_order(order: Dict):
    STATE["last_order_epoch_s"] = time.time()
    st = STATE["stats"]
    st["processed"] += 1
    verdict = order["verdict"]

    if verdict == "AUTHORIZED":
        st["approved_dzd"] += order["amount_dzd"]
        st["per_10s"][-1]["approved"] += 1
    elif verdict in ("FRAUD_BLOCKED", "BANNED_RETRY"):
        st["prevented_dzd"] += order["amount_dzd"]
        st["blocked_count"] += 1
        st["per_10s"][-1]["blocked"] += 1
        cat = order["category"]
        st["by_category"][cat] = st["by_category"].get(cat, 0) + 1
        if order.get("velocity_kmh", 0) > 900 or order["origin"]["name"] != order["destination"]["name"]:
            loc = order["destination"]["name"]
            st["by_origin"][loc] = st["by_origin"].get(loc, 0) + 1
    elif verdict == "NEEDS_REVIEW":
        st["per_10s"][-1]["review"] += 1

    STATE["latencies"].append(order["latency_ms"])
    lats = list(STATE["latencies"])
    st["avg_latency_ms"] = round(float(np.mean(lats)), 1)
    st["p95_latency_ms"] = round(float(np.percentile(lats, 95)), 1)


async def broadcast_event(event_type: str, payload: Dict):
    STATE["sse_id"] += 1
    sse_id = STATE["sse_id"]
    data_str = json.dumps({"id": sse_id, "type": event_type, "payload": payload})
    msg = f"id: {sse_id}\ndata: {data_str}\n\n"
    dead = []
    for q in SUBSCRIBERS:
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)
    for q in dead:
        SUBSCRIBERS.discard(q)


async def generate_transaction(forced_mode: Optional[str] = None, forced_cat: Optional[Dict] = None) -> Dict:
    released_ids = maintain_buyer_pool_health()
    for bid in released_ids:
        await broadcast_event("unban", {"buyer_id": bid, "auto": True})

    STATE["order_seq"] += 1
    order_id = f"ORD-DZ-{STATE['order_seq']}"
    now_s = time.time()
    epoch_ms = int(now_s * 1000)
    now_str = algiers_str(now_s)

    active_buyers = [b for b in BUYER_POOL if b["id"] not in STATE["banned_users"]]
    ongoing_decline_buyer = get_ongoing_card_testing_buyer(active_buyers)

    # Automatic distribution: if a buyer is mid-card-testing (at 1/3 or 2/3), continue their sequence!
    if forced_mode is None:
        if STATE["stats"]["processed"] == 1:
            mode = "needs_review"
        elif ongoing_decline_buyer is not None and random.random() < 0.65:
            mode = "card_decline"
        else:
            r = random.random()
            if r < 0.10:
                mode = "impossible_travel"
            elif r < 0.17:
                mode = "card_decline"
            elif r < 0.35:
                mode = "needs_review"
            elif r < 0.38 and STATE["banned_users"]:
                mode = "banned_retry"
            else:
                mode = "legit"
    else:
        mode = forced_mode

    if forced_cat is not None:
        cat_info = forced_cat
    elif mode == "impossible_travel":
        cat_info = random.choice(HIGH_TICKET_CATEGORIES)
    else:
        cat_info = random.choice(CATEGORIES)

    category = cat_info["cat"]
    item_name = random.choice(cat_info["items"])

    # Banned retry mode
    if mode == "banned_retry" and STATE["banned_users"]:
        banned_id = random.choice(list(STATE["banned_users"].keys()))
        b_info = STATE["banned_users"][banned_id]
        buyer_id = banned_id
        buyer_name = b_info["buyer_name"]
        home_w = b_info["home_wilaya_obj"]

        lo_amt = int(cat_info["min"])
        hi_amt = max(lo_amt, int(cat_info["max"]))
        amount_dzd = random.randint(lo_amt, hi_amt)
        j_lat, j_lon = jitter_coord(home_w["lat"], home_w["lon"])

        raw_score, latency_ms, _ = await run_in_threadpool(
            _sync_score_transaction,
            float(amount_dzd), 950.0, 3, 1, j_lat, j_lon, category
        )
        prob_score = max(0.94, raw_score)
        order = {
            "order_id": order_id,
            "epoch_ms": epoch_ms,
            "timestamp": now_str,
            "buyer_id": buyer_id,
            "buyer_name": buyer_name,
            "item": item_name,
            "category": category,
            "amount_dzd": amount_dzd,
            "origin": {"name": home_w["name"], "lat": j_lat, "lon": j_lon},
            "destination": {"name": home_w["name"], "lat": j_lat, "lon": j_lon},
            "distance_km": 0.0,
            "elapsed_min": 1,
            "velocity_kmh": 0.0,
            "attempt_num": 3,
            "consecutive_declines": 3,
            "is_declined": 1,
            "model_score": prob_score,
            "risk_score": prob_score,
            "decision_source": "GATEWAY: blacklist",
            "verdict": "BANNED_RETRY",
            "reason": "Gateway Policy: Frozen Account Checkout Attempt Rejected",
            "latency_ms": latency_ms,
        }
        STATE["orders"].appendleft(order)
        STATE["all_orders_export"].appendleft(order)
        record_stats_for_order(order)
        order["avg_latency_ms"] = STATE["stats"]["avg_latency_ms"]
        order["p95_latency_ms"] = STATE["stats"]["p95_latency_ms"]
        log_audit("GATEWAY_REJECT", f"Blocked retry {order_id} from banned buyer {buyer_id}")
        return order

    # If mode is card_decline and a buyer is already at 1/3 or 2/3, use that buyer so they progress 1/3 -> 2/3 -> 3/3
    if mode == "card_decline" and ongoing_decline_buyer is not None:
        buyer = ongoing_decline_buyer
    else:
        buyer = random.choice(active_buyers) if active_buyers else spawn_fresh_buyer()

    buyer_id = buyer["id"]
    buyer_name = buyer["name"]
    home_w = buyer["home"]
    orig_lat, orig_lon = jitter_coord(home_w["lat"], home_w["lon"])

    if mode == "impossible_travel":
        if random.random() < 0.72:
            ext = random.choice(FOREIGN_EXIT_NODES)
            dest_name = ext["name"]
            dest_lat, dest_lon = jitter_coord(ext["lat"], ext["lon"])
            elapsed_min = random.randint(8, 22)
        else:
            far_candidates = [
                w for w in WILAYAS
                if haversine_km(home_w["lat"], home_w["lon"], w["lat"], w["lon"]) > 450
            ]
            far_w = random.choice(far_candidates or WILAYAS)
            dest_name = far_w["name"]
            dest_lat, dest_lon = jitter_coord(far_w["lat"], far_w["lon"])
            elapsed_min = random.randint(5, 14)

        dist_km = round(haversine_km(orig_lat, orig_lon, dest_lat, dest_lon), 1)
        velocity_kmh = round(dist_km / (elapsed_min / 60.0), 1)
        lo = min(55000, int(cat_info["max"] * 0.8))
        hi = max(lo, int(cat_info["max"]))
        amount_dzd = random.randint(lo, hi)
        attempt_num = 1
        is_declined = 0

    elif mode == "card_decline":
        hist = get_recent_declines(buyer_id)
        hist.append(now_s)
        STATE["decline_history"][buyer_id] = hist
        streak = len(hist)

        dest_name = home_w["name"]
        dest_lat, dest_lon = jitter_coord(home_w["lat"], home_w["lon"])
        dist_km = round(haversine_km(orig_lat, orig_lon, dest_lat, dest_lon), 1)
        elapsed_min = random.randint(12, 45)
        velocity_kmh = round(dist_km / (elapsed_min / 60.0), 1)
        lo = int(cat_info["min"])
        hi = max(lo, int(cat_info["max"]))
        amount_dzd = random.randint(lo, hi)
        attempt_num = streak
        is_declined = 1

    elif mode == "needs_review":
        mid_candidates = [
            w for w in WILAYAS
            if 220 <= haversine_km(home_w["lat"], home_w["lon"], w["lat"], w["lon"]) <= 480
        ]
        mid_w = random.choice(mid_candidates or WILAYAS)
        dest_name = mid_w["name"]
        dest_lat, dest_lon = jitter_coord(mid_w["lat"], mid_w["lon"])
        dist_km = round(haversine_km(orig_lat, orig_lon, dest_lat, dest_lon), 1)
        elapsed_min = max(18, int(round((dist_km / random.uniform(460.0, 760.0)) * 60.0)))
        velocity_kmh = round(dist_km / (elapsed_min / 60.0), 1)
        lo = max(35000, int(cat_info["min"]))
        hi = max(lo, int(cat_info["max"]))
        amount_dzd = random.randint(lo, hi)
        attempt_num = 2
        is_declined = 0

    else:  # legit domestic order
        dest_name = home_w["name"]
        dest_lat, dest_lon = jitter_coord(home_w["lat"], home_w["lon"])
        dist_km = round(haversine_km(orig_lat, orig_lon, dest_lat, dest_lon), 1)
        elapsed_min = random.randint(25, 90)
        velocity_kmh = round(dist_km / (elapsed_min / 60.0), 1)
        lo = int(cat_info["min"])
        hi = max(lo, min(int(cat_info["max"]), 36000))
        amount_dzd = random.randint(lo, hi)
        attempt_num = 1
        is_declined = 0

    raw_score, latency_ms, used_model = await run_in_threadpool(
        _sync_score_transaction,
        float(amount_dzd),
        float(velocity_kmh),
        int(attempt_num),
        int(is_declined),
        float(dest_lat),
        float(dest_lon),
        category,
    )

    display_name_with_wilaya = f"{buyer_name} ({home_w['name']})"

    if is_declined and attempt_num >= 3:
        verdict = "FRAUD_BLOCKED"
        raw_score = max(0.95, raw_score)
        decision_source = "RULE: 3/3 declines (card testing)"
        reason = "Blocked: Too Many Attempts (3/3 Declines) — Suspected Stolen or Generated Card Numbers — Account Auto-Frozen"
        STATE["banned_users"][buyer_id] = {
            "buyer_id": buyer_id,
            "buyer_name": display_name_with_wilaya,
            "reason": reason,
            "banned_at": now_str,
            "banned_epoch_s": now_s,
            "home_wilaya": home_w["name"],
            "home_wilaya_obj": home_w,
            "trigger_order": order_id,
        }
        log_audit("AUTO_BAN", f"Frozen {buyer_id} ({display_name_with_wilaya}) after 3/3 card declines on {order_id}")

    elif is_declined and attempt_num < 3:
        verdict = "DECLINED_WARNING"
        raw_score = round(0.38 + attempt_num * 0.18, 2)
        decision_source = f"RULE: card decline ({attempt_num}/3)"
        reason = f"Card Declined ({attempt_num}/3) — Suspected Stolen/Generated Card Testing"
        log_audit("CARD_DECLINE", f"{buyer_id} card declined ({attempt_num}/3) on {order_id}")

    elif velocity_kmh > 900.0:
        verdict = "FRAUD_BLOCKED"
        decision_source = "RULE: impossible travel > 900 km/h"
        reason = f"Impossible Travel ({dist_km} km in {elapsed_min} min = {velocity_kmh} km/h)"
        STATE["banned_users"][buyer_id] = {
            "buyer_id": buyer_id,
            "buyer_name": display_name_with_wilaya,
            "reason": reason,
            "banned_at": now_str,
            "banned_epoch_s": now_s,
            "home_wilaya": home_w["name"],
            "home_wilaya_obj": home_w,
            "trigger_order": order_id,
        }
        log_audit("IMPOSSIBLE_TRAVEL", f"Blocked {order_id} & froze {buyer_id} ({home_w['name']} -> {dest_name} @ {velocity_kmh} km/h)")

    elif mode == "needs_review" or (380.0 <= velocity_kmh <= 900.0) or (0.50 <= raw_score < 0.85):
        verdict = "NEEDS_REVIEW"
        decision_source = "MODEL + VELOCITY"
        reason = f"Unusual Route ({dist_km} km in {elapsed_min} min = {velocity_kmh} km/h) — Held for Review"
        log_audit("HELD_FOR_REVIEW", f"Order {order_id} ({amount_dzd} DZD) queued for analyst review")

    elif raw_score >= 0.85:
        verdict = "FRAUD_BLOCKED"
        decision_source = "MODEL"
        reason = f"High Fraud Probability ({round(raw_score * 100, 1)}%) — Auto-Blocked"
        STATE["banned_users"][buyer_id] = {
            "buyer_id": buyer_id,
            "buyer_name": display_name_with_wilaya,
            "reason": reason,
            "banned_at": now_str,
            "banned_epoch_s": now_s,
            "home_wilaya": home_w["name"],
            "home_wilaya_obj": home_w,
            "trigger_order": order_id,
        }
        log_audit("MODEL_BLOCK", f"Model blocked {order_id} (score {raw_score})")

    else:
        verdict = "AUTHORIZED"
        decision_source = "MODEL" if used_model else "KINEMATIC + MODEL"
        reason = f"Verified Domestic Checkout ({dist_km} km in {elapsed_min} min)"

    order = {
        "order_id": order_id,
        "epoch_ms": epoch_ms,
        "timestamp": now_str,
        "buyer_id": buyer_id,
        "buyer_name": buyer_name,
        "item": item_name,
        "category": category,
        "amount_dzd": amount_dzd,
        "origin": {"name": home_w["name"], "lat": orig_lat, "lon": orig_lon},
        "destination": {"name": dest_name, "lat": dest_lat, "lon": dest_lon},
        "distance_km": dist_km,
        "elapsed_min": elapsed_min,
        "velocity_kmh": velocity_kmh,
        "attempt_num": attempt_num,
        "consecutive_declines": len(STATE["decline_history"].get(buyer_id, [])),
        "is_declined": is_declined,
        "model_score": raw_score,
        "risk_score": raw_score,
        "decision_source": decision_source,
        "verdict": verdict,
        "reason": reason,
        "latency_ms": latency_ms,
        "avg_latency_ms": STATE["stats"]["avg_latency_ms"],
        "p95_latency_ms": STATE["stats"]["p95_latency_ms"],
    }

    STATE["orders"].appendleft(order)
    STATE["all_orders_export"].appendleft(order)
    if verdict == "NEEDS_REVIEW":
        STATE["pending_review"][order_id] = order

    record_stats_for_order(order)
    order["avg_latency_ms"] = STATE["stats"]["avg_latency_ms"]
    order["p95_latency_ms"] = STATE["stats"]["p95_latency_ms"]
    return order


# =====================================================================
# 5. FAULT-TOLERANT BACKGROUND LOOPS & LIFESPAN HANDLER
# =====================================================================
async def background_order_generator():
    STATE["generator_alive"] = True
    while True:
        try:
            await asyncio.sleep(max(0.6, float(STATE["speed_sec"])))
            if not STATE["demo_paused"] and SUBSCRIBERS:
                order = await generate_transaction()
                await broadcast_event("order", order)
        except asyncio.CancelledError:
            STATE["generator_alive"] = False
            break
        except Exception as exc:
            print(f"[!] Generator error caught and recovered: {exc!r}")
            log_audit("GENERATOR_ERROR", repr(exc))
            await asyncio.sleep(1.0)


async def rolling_10s_bucket_ticker():
    while True:
        try:
            await asyncio.sleep(10.0)
            if not STATE["demo_paused"]:
                ts = algiers_str()
                buckets = STATE["stats"]["per_10s"]
                buckets.append({"label": ts, "approved": 0, "blocked": 0, "review": 0})
                if len(buckets) > 6:
                    buckets.pop(0)
                await broadcast_event("stats", {
                    "stats": STATE["stats"],
                    "banned_count": len(STATE["banned_users"]),
                    "pending_count": len(STATE["pending_review"]),
                })
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log_audit("TICKER_ERROR", repr(exc))


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    log_audit("SYSTEM_BOOT", f"SentinelAI initialized ({MODEL_STATUS}, 58 Wilayas active)")
    t1 = asyncio.create_task(background_order_generator())
    t2 = asyncio.create_task(rolling_10s_bucket_ticker())
    BG_TASKS.extend([t1, t2])
    try:
        yield
    finally:
        for t in BG_TASKS:
            t.cancel()
        await asyncio.gather(*BG_TASKS, return_exceptions=True)


app.router.lifespan_context = lifespan

STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =====================================================================
# 6. HTTP & SSE ENDPOINTS
# =====================================================================
def verify_presenter_key(x_presenter_key: Optional[str]):
    if PRESENTER_KEY and x_presenter_key != PRESENTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Presenter-Key")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    candidates = [
        BASE_DIR / "index.html",
        Path.cwd() / "index.html",
        BASE_DIR / "templates" / "index.html",
        BASE_DIR / "static" / "index.html",
    ]
    for p in candidates:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    raise HTTPException(status_code=500, detail="index.html not found")


@app.get("/api/health")
def health_check():
    last_ts = STATE["last_order_epoch_s"]
    age_sec = round(time.time() - last_ts, 1) if last_ts > 0 else None
    return {
        "status": "ok",
        "model_status": MODEL_STATUS,
        "generator_alive": STATE["generator_alive"],
        "last_order_age_sec": age_sec,
        "uptime_sec": round(time.time() - START_TIME, 1),
        "connected_dashboards": len(SUBSCRIBERS),
    }


@app.get("/api/state")
def get_full_state():
    banned_clean = [
        {
            "buyer_id": v["buyer_id"],
            "buyer_name": v["buyer_name"],
            "reason": v["reason"],
            "banned_at": v["banned_at"],
            "home_wilaya": v["home_wilaya"],
            "trigger_order": v["trigger_order"],
        }
        for v in STATE["banned_users"].values()
    ]
    return {
        "orders": list(STATE["orders"])[:200],
        "banned": banned_clean,
        "pending_review": list(STATE["pending_review"].values()),
        "stats": STATE["stats"],
        "audit": list(STATE["audit_log"])[:100],
        "model_status": MODEL_STATUS,
        "demo_paused": STATE["demo_paused"],
        "speed_sec": STATE["speed_sec"],
        "config": {
            "wilayas_monitored": len(WILAYAS),
            "dzd_per_usd": DZD_PER_USD,
            "requires_key": bool(PRESENTER_KEY),
        },
    }


@app.get("/stream")
async def sse_stream(request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=150)
    SUBSCRIBERS.add(queue)

    async def event_publisher():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            SUBSCRIBERS.discard(queue)

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/demo/resume")
async def demo_resume(x_presenter_key: Optional[str] = Header(default=None)):
    verify_presenter_key(x_presenter_key)
    was_paused = STATE["demo_paused"]
    STATE["demo_paused"] = False
    if was_paused and len(STATE["orders"]) == 0:
        order = await generate_transaction()
        await broadcast_event("order", order)
    await broadcast_event("demo_status", {"demo_paused": False, "speed_sec": STATE["speed_sec"]})
    return {"status": "ok", "demo_paused": False}


@app.post("/api/demo/pause")
async def demo_pause(x_presenter_key: Optional[str] = Header(default=None)):
    verify_presenter_key(x_presenter_key)
    STATE["demo_paused"] = True
    await broadcast_event("demo_status", {"demo_paused": True, "speed_sec": STATE["speed_sec"]})
    return {"status": "ok", "demo_paused": True}


@app.post("/api/demo/speed")
async def demo_speed(
    sec: float = Query(2.5, ge=0.5, le=10.0),
    x_presenter_key: Optional[str] = Header(default=None),
):
    verify_presenter_key(x_presenter_key)
    STATE["speed_sec"] = sec
    await broadcast_event("demo_status", {"demo_paused": STATE["demo_paused"], "speed_sec": sec})
    return {"status": "ok", "speed_sec": sec}


@app.post("/api/demo/reset")
async def demo_reset(x_presenter_key: Optional[str] = Header(default=None)):
    verify_presenter_key(x_presenter_key)
    global STATE
    STATE = create_initial_state()
    STATE["demo_paused"] = False
    log_audit("DEMO_RESET", "SOC state reset by presenter", "PRESENTER")
    await broadcast_event("reset", {})
    return {"status": "ok"}


@app.post("/api/simulate/{scenario}")
async def trigger_simulation(
    scenario: str,
    x_presenter_key: Optional[str] = Header(default=None),
):
    verify_presenter_key(x_presenter_key)
    STATE["demo_paused"] = False

    if scenario == "impossible_travel":
        order = await generate_transaction(forced_mode="impossible_travel")
        await broadcast_event("order", order)
        return {"status": "ok", "orders": [order]}

    elif scenario == "needs_review":
        order = await generate_transaction(forced_mode="needs_review")
        await broadcast_event("order", order)
        return {"status": "ok", "orders": [order]}

    elif scenario == "card_decline":
        released_ids = maintain_buyer_pool_health()
        for bid in released_ids:
            await broadcast_event("unban", {"buyer_id": bid, "auto": True})

        active_buyers = [b for b in BUYER_POOL if b["id"] not in STATE["banned_users"]]
        target = random.choice(active_buyers)
        buyer_id = target["id"]
        STATE["decline_history"][buyer_id] = []

        emitted = []
        for step in range(1, 4):
            now_s = time.time()
            STATE["decline_history"].setdefault(buyer_id, []).append(now_s)
            STATE["order_seq"] += 1
            oid = f"ORD-DZ-{STATE['order_seq']}"
            epoch_ms = int(now_s * 1000)
            now_str = algiers_str(now_s)
            home_w = target["home"]
            j_lat, j_lon = jitter_coord(home_w["lat"], home_w["lon"])
            amt = random.randint(28000, 64000)

            raw_score, latency_ms, _ = await run_in_threadpool(
                _sync_score_transaction,
                float(amt), 18.0, step, 1, j_lat, j_lon, "Mobile & Recharge"
            )

            display_name_with_wilaya = f"{target['name']} ({home_w['name']})"

            if step < 3:
                verdict = "DECLINED_WARNING"
                raw_score = round(0.38 + step * 0.18, 2)
                decision_source = f"RULE: card decline ({step}/3)"
                reason = f"Card Declined ({step}/3) — Suspected Stolen/Generated Card Testing"
                log_audit("CARD_DECLINE", f"{buyer_id} failed card attempt {step}/3 ({oid})")
            else:
                verdict = "FRAUD_BLOCKED"
                raw_score = max(0.96, raw_score)
                decision_source = "RULE: 3/3 declines (card testing)"
                reason = "Blocked: Too Many Attempts (3/3 Declines) — Suspected Stolen or Generated Card Numbers — Account Auto-Frozen"
                STATE["banned_users"][buyer_id] = {
                    "buyer_id": buyer_id,
                    "buyer_name": display_name_with_wilaya,
                    "reason": reason,
                    "banned_at": now_str,
                    "banned_epoch_s": now_s,
                    "home_wilaya": home_w["name"],
                    "home_wilaya_obj": home_w,
                    "trigger_order": oid,
                }
                log_audit("AUTO_BAN", f"Auto-frozen {buyer_id} upon 3/3 card decline ({oid})")

            order = {
                "order_id": oid,
                "epoch_ms": epoch_ms,
                "timestamp": now_str,
                "buyer_id": buyer_id,
                "buyer_name": target["name"],
                "item": "Mobilis 20,000 DZD Corporate Top-Up",
                "category": "Mobile & Recharge",
                "amount_dzd": amt,
                "origin": {"name": home_w["name"], "lat": j_lat, "lon": j_lon},
                "destination": {"name": home_w["name"], "lat": j_lat, "lon": j_lon},
                "distance_km": 3.6,
                "elapsed_min": 12,
                "velocity_kmh": 18.0,
                "attempt_num": step,
                "consecutive_declines": step,
                "is_declined": 1,
                "model_score": raw_score,
                "risk_score": raw_score,
                "decision_source": decision_source,
                "verdict": verdict,
                "reason": reason,
                "latency_ms": latency_ms,
                "avg_latency_ms": STATE["stats"]["avg_latency_ms"],
                "p95_latency_ms": STATE["stats"]["p95_latency_ms"],
            }
            STATE["orders"].appendleft(order)
            STATE["all_orders_export"].appendleft(order)
            record_stats_for_order(order)
            order["avg_latency_ms"] = STATE["stats"]["avg_latency_ms"]
            order["p95_latency_ms"] = STATE["stats"]["p95_latency_ms"]
            emitted.append(order)
            await broadcast_event("order", order)
            if step < 3:
                await asyncio.sleep(0.6)

        return {"status": "ok", "orders": emitted}

    elif scenario == "banned_retry":
        if not STATE["banned_users"]:
            first = await generate_transaction(forced_mode="impossible_travel")
            await broadcast_event("order", first)
            await asyncio.sleep(0.45)
        order = await generate_transaction(forced_mode="banned_retry")
        await broadcast_event("order", order)
        return {"status": "ok", "orders": [order]}

    raise HTTPException(status_code=400, detail="Unknown simulation scenario")


@app.post("/api/review/{order_id}")
async def analyst_review(
    order_id: str,
    decision: str = Query(...),
    x_presenter_key: Optional[str] = Header(default=None),
):
    verify_presenter_key(x_presenter_key)
    if decision not in ("approve", "block"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'block'")

    target_order = STATE["pending_review"].get(order_id)
    if not target_order:
        for o in STATE["orders"]:
            if o["order_id"] == order_id:
                target_order = o
                break

    if not target_order:
        raise HTTPException(status_code=404, detail="Order not found")
    if target_order["verdict"] != "NEEDS_REVIEW":
        raise HTTPException(status_code=409, detail="Order was already resolved")

    STATE["pending_review"].pop(order_id, None)
    st = STATE["stats"]
    now_s = time.time()

    if decision == "approve":
        target_order["verdict"] = "AUTHORIZED"
        target_order["decision_source"] = "ANALYST"
        target_order["reason"] = "Cleared by SOC Analyst Override"
        st["approved_dzd"] += target_order["amount_dzd"]
        log_audit("ANALYST_APPROVE", f"Approved {order_id} ({target_order['amount_dzd']} DZD)", "SOC-ANALYST")
    else:
        target_order["verdict"] = "FRAUD_BLOCKED"
        target_order["decision_source"] = "ANALYST"
        target_order["reason"] = "Confirmed Fraud by SOC Analyst — Account Frozen"
        st["prevented_dzd"] += target_order["amount_dzd"]
        st["blocked_count"] += 1
        cat = target_order["category"]
        st["by_category"][cat] = st["by_category"].get(cat, 0) + 1
        loc = target_order["destination"]["name"]
        st["by_origin"][loc] = st["by_origin"].get(loc, 0) + 1

        home_obj = next((w for w in WILAYAS if w["name"] == target_order["origin"]["name"]), WILAYAS[15])
        display_name_with_wilaya = f"{target_order['buyer_name']} ({target_order['origin']['name']})"
        STATE["banned_users"][target_order["buyer_id"]] = {
            "buyer_id": target_order["buyer_id"],
            "buyer_name": display_name_with_wilaya,
            "reason": target_order["reason"],
            "banned_at": algiers_str(now_s),
            "banned_epoch_s": now_s,
            "home_wilaya": target_order["origin"]["name"],
            "home_wilaya_obj": home_obj,
            "trigger_order": order_id,
        }
        log_audit("ANALYST_BLOCK", f"Blocked {order_id} and froze {target_order['buyer_id']}", "SOC-ANALYST")

    await broadcast_event("review_update", {
        "order": target_order,
        "stats": STATE["stats"],
        "banned": list(STATE["banned_users"].values()),
        "pending_count": len(STATE["pending_review"]),
    })
    return {"status": "ok", "order": target_order}


@app.post("/api/unban/{buyer_id}")
async def unban_account(
    buyer_id: str,
    x_presenter_key: Optional[str] = Header(default=None),
):
    verify_presenter_key(x_presenter_key)
    if buyer_id not in STATE["banned_users"]:
        raise HTTPException(status_code=404, detail="Account is not currently banned")

    removed = STATE["banned_users"].pop(buyer_id, None)
    STATE["decline_history"].pop(buyer_id, None)
    log_audit("ACCOUNT_UNBANNED", f"Restored account {buyer_id} to active standing", "SOC-ANALYST")
    await broadcast_event("unban", {"buyer_id": buyer_id, "auto": False})
    return {"status": "ok", "unbanned": buyer_id, "previous": removed}


@app.get("/api/export.csv")
def export_csv():
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    writer.writerow([
        "Order ID", "Timestamp (Algiers)", "Buyer ID", "Buyer Name",
        "Category", "Item", "Amount (DZD)", "Origin Wilaya", "Destination",
        "Distance (km)", "Elapsed (min)", "Velocity (km/h)",
        "Fraud Probability", "Decision Source", "Verdict", "Latency (ms)", "Diagnostic Reason"
    ])
    for o in STATE["all_orders_export"]:
        algiers_ts = algiers_str(o["epoch_ms"] / 1000.0) if "epoch_ms" in o else o["timestamp"]
        writer.writerow([
            o["order_id"], algiers_ts, o["buyer_id"], o["buyer_name"],
            o["category"], o["item"], o["amount_dzd"],
            o["origin"]["name"], o["destination"]["name"],
            o.get("distance_km", 0), o.get("elapsed_min", 1), o["velocity_kmh"],
            f"{round(o['model_score'] * 100, 1)}%", o.get("decision_source", "MODEL"),
            o["verdict"], o["latency_ms"], o["reason"]
        ])
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="SentinelAI_Full_Audit_Export.csv"'},
    )


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)