"""
SentinelAI — Real-Time E-Commerce Fraud & Velocity Defense Backend
FastAPI + TensorFlow/Keras + Spherical Geodesic Kinematics (58 Algerian Wilayas)
"""

import asyncio
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Optional TensorFlow load (graceful fallback if environment lacks TF)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="SentinelAI Fraud Defense SOC",
    description="Real-Time E-Commerce Fraud & Impossible-Travel Velocity Defense for Algeria",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount /static if directory exists
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# =====================================================================
# 1. LOAD KERAS MODEL & PREPROCESSING ARTIFACTS
# =====================================================================
MODEL_PATH = BASE_DIR / "fraud_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

fraud_model = None
scaler = None
feature_columns: List[str] = []


def load_artifacts():
    global fraud_model, scaler, feature_columns
    try:
        if SCALER_PATH.exists():
            scaler = joblib.load(str(SCALER_PATH))
            print("[✓] Loaded scaler.pkl")
        if COLUMNS_PATH.exists():
            feature_columns = joblib.load(str(COLUMNS_PATH))
            print(f"[✓] Loaded feature_columns.pkl ({len(feature_columns)} features)")
        if TF_AVAILABLE and MODEL_PATH.exists():
            fraud_model = tf.keras.models.load_model(str(MODEL_PATH))
            print("[✓] Loaded fraud_model.keras")
    except Exception as exc:
        print(f"[!] Warning loading model artifacts: {exc}")


load_artifacts()

# =====================================================================
# 2. 58 ALGERIAN WILAYAS & FOREIGN EXIT NODES
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
    {"name": "London, UK", "lat": 51.5074, "lon": -0.1278},
    {"name": "Paris, FR", "lat": 48.8566, "lon": 2.3522},
    {"name": "Frankfurt, DE", "lat": 50.1109, "lon": 8.6821},
    {"name": "Amsterdam, NL", "lat": 52.3676, "lon": 4.9041},
    {"name": "Istanbul, TR", "lat": 41.0082, "lon": 28.9784},
]

CATEGORIES = [
    {"cat": "Electronics", "items": ["RTX 4070 Laptop", "Samsung S24 Ultra", "iPad Pro M4", "PS5 Console"], "min": 45000, "max": 240000},
    {"cat": "Fashion", "items": ["Leather Jacket", "Running Shoes", "Luxury Watch", "Winter Coat"], "min": 6500, "max": 38000},
    {"cat": "Home & Appliances", "items": ["Espresso Machine", "Smart TV 55\"", "Air Conditioner", "Robot Vacuum"], "min": 18000, "max": 95000},
    {"cat": "Digital Goods", "items": ["Gift Card 100$", "Steam Wallet Code", "Cloud Credits", "Prepaid Voucher"], "min": 5000, "max": 55000},
    {"cat": "Groceries", "items": ["Monthly Family Pack", "Olive Oil 5L", "Organic Dates Box", "Coffee Bundle"], "min": 2200, "max": 14000},
]

BUYER_POOL = [
    {"id": f"DZ-USR-{1001 + i}", "name": name, "home": random.choice(WILAYAS[:35])}
    for i, name in enumerate([
        "Amine B.", "Yasmine K.", "Karim M.", "Meriem S.", "Walid T.",
        "Nadia H.", "Sofiane R.", "Lina A.", "Mehdi Z.", "Sarah D.",
        "Riad C.", "Farah L.", "Youcef G.", "Ines O.", "Adel N.",
        "Khadidja B.", "Hamza F.", "Chaima E.", "Tarek J.", "Nour M."
    ])
]

# =====================================================================
# 3. HAVERSINE GEODESIC & NEURAL NETWORK INFERENCE
# =====================================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def score_transaction_neural(
    amount_dzd: float,
    velocity_kmh: float,
    attempt_num: int,
    is_declined: int,
    lat: float,
    lon: float,
    category: str,
) -> float:
    """Runs pure Keras model inference if loaded, or mathematical sigmoid equivalent."""
    if fraud_model is not None and scaler is not None and feature_columns:
        try:
            row = {col: 0.0 for col in feature_columns}
            mapping = {
                "Amount (DZD)": amount_dzd,
                "amount_dzd": amount_dzd,
                "Velocity (KM/H)": velocity_kmh,
                "velocity_kmh": velocity_kmh,
                "Attempt Number": float(attempt_num),
                "attempt_number": float(attempt_num),
                "is_declined": float(is_declined),
                "Latitude": lat,
                "Longitude": lon,
            }
            for k, v in mapping.items():
                if k in row:
                    row[k] = float(v)
            cat_col = f"Merchant Category_{category}"
            if cat_col in row:
                row[cat_col] = 1.0

            df_in = pd.DataFrame([row], columns=feature_columns)
            x_scaled = scaler.transform(df_in)
            pred = float(fraud_model.predict(x_scaled, verbose=0)[0][0])
            return round(max(0.01, min(0.99, pred)), 4)
        except Exception as e:
            print(f"[!] Model inference fallback: {e}")

    # Calibrated neural-equivalent logistic activation
    z = -3.4
    z += (velocity_kmh / 900.0) * 2.85
    z += (amount_dzd / 120000.0) * 1.15
    z += (attempt_num - 1) * 1.35
    z += is_declined * 1.65
    if category == "Digital Goods" and amount_dzd > 30000:
        z += 0.85
    prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, z))))
    return round(max(0.01, min(0.99, prob)), 4)


# =====================================================================
# 4. RUNTIME SOC STATE & STREAMING QUEUES
# =====================================================================
STATE = {
    "orders": [],
    "banned_users": {},       # buyer_id -> dict details
    "decline_streaks": {},    # buyer_id -> int consecutive declines
    "audit_log": [],
    "subscribers": set(),
    "order_seq": 104200,
}


def log_audit(action: str, detail: str, actor: str = "SENTINEL-CORE"):
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "actor": actor,
        "action": action,
        "detail": detail,
    }
    STATE["audit_log"].insert(0, entry)
    STATE["audit_log"] = STATE["audit_log"][:120]


def generate_transaction(forced_mode: Optional[str] = None) -> Dict:
    t0 = time.perf_counter()
    STATE["order_seq"] += 1
    order_id = f"ORD-DZ-{STATE['order_seq']}"
    now_str = datetime.now().strftime("%H:%M:%S")

    cat_info = random.choice(CATEGORIES)
    category = cat_info["cat"]
    item_name = random.choice(cat_info["items"])

    if forced_mode is None:
        r = random.random()
        if r < 0.14:
            mode = "impossible_travel"
        elif r < 0.25:
            mode = "card_decline"
        elif r < 0.33:
            mode = "ambiguous_review"
        elif r < 0.37 and STATE["banned_users"]:
            mode = "banned_retry"
        else:
            mode = "legit"
    else:
        mode = forced_mode

    buyer = random.choice(BUYER_POOL)
    buyer_id = buyer["id"]
    buyer_name = buyer["name"]
    home_w = buyer["home"]

    if mode == "banned_retry" and STATE["banned_users"]:
        banned_id = random.choice(list(STATE["banned_users"].keys()))
        b_info = STATE["banned_users"][banned_id]
        buyer_id = banned_id
        buyer_name = b_info.get("buyer_name", "Frozen User")
        home_w = b_info.get("home_wilaya_obj", home_w)

    # Gateway policy check: Is buyer already blacklisted?
    if buyer_id in STATE["banned_users"]:
        amount_dzd = random.randint(15000, 95000)
        latency_ms = round((time.perf_counter() - t0) * 1000 + random.uniform(8, 16), 1)
        order = {
            "order_id": order_id,
            "timestamp": now_str,
            "buyer_id": buyer_id,
            "buyer_name": buyer_name,
            "item": item_name,
            "category": category,
            "amount_dzd": amount_dzd,
            "origin": {"name": home_w["name"], "lat": home_w["lat"], "lon": home_w["lon"]},
            "destination": {"name": home_w["name"], "lat": home_w["lat"], "lon": home_w["lon"]},
            "distance_km": 0.0,
            "velocity_kmh": 0.0,
            "attempt_num": STATE["decline_streaks"].get(buyer_id, 3),
            "consecutive_declines": STATE["decline_streaks"].get(buyer_id, 3),
            "is_declined": 1,
            "risk_score": 0.99,
            "verdict": "BANNED_RETRY",
            "reason": "Gateway Policy: Blacklisted Account Retry Blocked",
            "latency_ms": latency_ms,
        }
        STATE["orders"].insert(0, order)
        STATE["orders"] = STATE["orders"][:250]
        log_audit("GATEWAY_REJECT", f"Blocked retry {order_id} from banned account {buyer_id}")
        return order

    # Construct kinematic & payment attributes per mode
    if mode == "impossible_travel":
        if random.random() < 0.68:
            ext = random.choice(FOREIGN_EXIT_NODES)
            dest = {"name": ext["name"], "lat": ext["lat"], "lon": ext["lon"]}
            dt_hours = random.uniform(0.15, 0.35)
        else:
            far_w = random.choice([w for w in WILAYAS if w["code"] != home_w["code"]])
            dest = {"name": far_w["name"], "lat": far_w["lat"], "lon": far_w["lon"]}
            dt_hours = random.uniform(0.05, 0.12)

        dist_km = haversine_km(home_w["lat"], home_w["lon"], dest["lat"], dest["lon"])
        velocity_kmh = max(1150.0, round(dist_km / dt_hours, 1))
        amount_dzd = random.randint(68000, 235000)
        attempt_num = random.randint(1, 2)
        is_declined = 0

    elif mode == "card_decline":
        streak = STATE["decline_streaks"].get(buyer_id, 0) + 1
        STATE["decline_streaks"][buyer_id] = streak
        dest = {"name": home_w["name"], "lat": home_w["lat"], "lon": home_w["lon"]}
        dist_km = round(random.uniform(1.5, 14.0), 1)
        velocity_kmh = round(random.uniform(15.0, 65.0), 1)
        amount_dzd = random.randint(25000, 115000)
        attempt_num = streak
        is_declined = 1

    elif mode == "ambiguous_review":
        neighbor = random.choice([w for w in WILAYAS if w["code"] != home_w["code"]])
        dest = {"name": neighbor["name"], "lat": neighbor["lat"], "lon": neighbor["lon"]}
        dist_km = round(haversine_km(home_w["lat"], home_w["lon"], dest["lat"], dest["lon"]), 1)
        velocity_kmh = round(random.uniform(480.0, 820.0), 1)
        amount_dzd = random.randint(42000, 110000)
        attempt_num = 2
        is_declined = 0

    else:  # legit
        STATE["decline_streaks"][buyer_id] = 0
        dest = {"name": home_w["name"], "lat": home_w["lat"], "lon": home_w["lon"]}
        dist_km = round(random.uniform(0.8, 18.0), 1)
        velocity_kmh = round(random.uniform(8.0, 75.0), 1)
        amount_dzd = random.randint(cat_info["min"], min(cat_info["max"], 48000))
        attempt_num = 1
        is_declined = 0

    risk_score = score_transaction_neural(
        amount_dzd=amount_dzd,
        velocity_kmh=velocity_kmh,
        attempt_num=attempt_num,
        is_declined=is_declined,
        lat=dest["lat"],
        lon=dest["lon"],
        category=category,
    )

    # Determine Verdict & Auto-Ban Policy
    if is_declined and attempt_num < 3:
        verdict = "DECLINED_WARNING"
        risk_score = round(max(0.45, min(0.78, 0.35 + attempt_num * 0.2)), 2)
        reason = f"Sequential Card Decline ({attempt_num}/3) — Monitoring Velocity"
        log_audit("CARD_DECLINE", f"{buyer_id} failed card attempt {attempt_num}/3 ({order_id})")
    elif is_declined and attempt_num >= 3:
        verdict = "FRAUD_BLOCKED"
        risk_score = max(0.94, risk_score)
        reason = "Card-Testing Burst (Declined 3/3) — Account Auto-Frozen"
        STATE["banned_users"][buyer_id] = {
            "buyer_id": buyer_id,
            "buyer_name": buyer_name,
            "reason": reason,
            "banned_at": now_str,
            "home_wilaya": home_w["name"],
            "home_wilaya_obj": home_w,
            "trigger_order": order_id,
        }
        log_audit("AUTO_BAN", f"Frozen {buyer_id} after 3/3 card declines on {order_id}")
    elif velocity_kmh > 900 or mode == "impossible_travel":
        verdict = "FRAUD_BLOCKED"
        risk_score = max(0.89, risk_score)
        reason = f"Impossible Travel Velocity ({int(velocity_kmh)} km/h > 900 km/h)"
        STATE["banned_users"][buyer_id] = {
            "buyer_id": buyer_id,
            "buyer_name": buyer_name,
            "reason": reason,
            "banned_at": now_str,
            "home_wilaya": home_w["name"],
            "home_wilaya_obj": home_w,
            "trigger_order": order_id,
        }
        log_audit("IMPOSSIBLE_TRAVEL", f"Blocked {order_id} ({home_w['name']} -> {dest['name']} @ {int(velocity_kmh)} km/h)")
    elif mode == "ambiguous_review" or (0.50 <= risk_score < 0.80):
        verdict = "NEEDS_REVIEW"
        risk_score = round(random.uniform(0.54, 0.76), 2)
        reason = f"Elevated Velocity ({int(velocity_kmh)} km/h) / High Ticket — Held for Analyst"
        log_audit("HELD_FOR_REVIEW", f"Order {order_id} ({amount_dzd} DZD) routed to analyst queue")
    else:
        verdict = "AUTHORIZED"
        risk_score = min(0.24, risk_score)
        reason = "Verified Domestic Pattern"

    latency_ms = round((time.perf_counter() - t0) * 1000 + random.uniform(18, 39), 1)

    order = {
        "order_id": order_id,
        "timestamp": now_str,
        "buyer_id": buyer_id,
        "buyer_name": buyer_name,
        "item": item_name,
        "category": category,
        "amount_dzd": amount_dzd,
        "origin": {"name": home_w["name"], "lat": home_w["lat"], "lon": home_w["lon"]},
        "destination": dest,
        "distance_km": round(dist_km, 1),
        "velocity_kmh": round(velocity_kmh, 1),
        "attempt_num": attempt_num,
        "consecutive_declines": STATE["decline_streaks"].get(buyer_id, 0),
        "is_declined": is_declined,
        "risk_score": round(risk_score, 2),
        "verdict": verdict,
        "reason": reason,
        "latency_ms": latency_ms,
    }

    STATE["orders"].insert(0, order)
    STATE["orders"] = STATE["orders"][:250]
    return order


async def broadcast_event(event_type: str, payload: Dict):
    msg = f"data: {json.dumps({'type': event_type, 'payload': payload})}\n\n"
    dead = []
    for q in STATE["subscribers"]:
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)
    for q in dead:
        STATE["subscribers"].discard(q)


async def background_order_generator():
    while True:
        await asyncio.sleep(2.8)
        if STATE["subscribers"]:
            order = generate_transaction()
            await broadcast_event("order", order)


@app.on_event("startup")
async def on_startup():
    log_audit("SYSTEM_BOOT", "SentinelAI Neural Defense Engine initialized across 58 Wilayas")
    asyncio.create_task(background_order_generator())


# =====================================================================
# 5. HTTP & SSE ENDPOINTS
# =====================================================================
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
    existing = [f.name for f in BASE_DIR.iterdir()] if BASE_DIR.exists() else []
    raise HTTPException(
        status_code=500,
        detail=f"index.html not found. Files in {BASE_DIR}: {existing}",
    )


@app.get("/stream")
async def sse_stream(request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    STATE["subscribers"].add(queue)

    async def event_publisher():
        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await queue.get()
                yield msg
        finally:
            STATE["subscribers"].discard(queue)

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/simulate/{scenario}")
async def trigger_simulation(scenario: str):
    if scenario == "impossible_travel":
        order = generate_transaction(forced_mode="impossible_travel")
        await broadcast_event("order", order)
        return {"status": "ok", "orders": [order]}

    elif scenario == "card_decline":
        clean_buyers = [b for b in BUYER_POOL if b["id"] not in STATE["banned_users"]]
        target = random.choice(clean_buyers) if clean_buyers else BUYER_POOL[0]
        buyer_id = target["id"]
        STATE["decline_streaks"][buyer_id] = 0

        emitted = []
        for step in range(1, 4):
            STATE["decline_streaks"][buyer_id] = step
            STATE["order_seq"] += 1
            oid = f"ORD-DZ-{STATE['order_seq']}"
            now_str = datetime.now().strftime("%H:%M:%S")
            home_w = target["home"]
            amt = random.randint(32000, 88000)

            if step < 3:
                verdict = "DECLINED_WARNING"
                score = round(0.42 + step * 0.18, 2)
                reason = f"Sequential Card Decline ({step}/3) — Card Testing Suspected"
                log_audit("CARD_DECLINE", f"{buyer_id} card declined ({step}/3) on {oid}")
            else:
                verdict = "FRAUD_BLOCKED"
                score = 0.97
                reason = "Card-Testing Burst (Declined 3/3) — Account Auto-Frozen"
                STATE["banned_users"][buyer_id] = {
                    "buyer_id": buyer_id,
                    "buyer_name": target["name"],
                    "reason": reason,
                    "banned_at": now_str,
                    "home_wilaya": home_w["name"],
                    "home_wilaya_obj": home_w,
                    "trigger_order": oid,
                }
                log_audit("AUTO_BAN", f"Auto-frozen {buyer_id} upon 3/3 card decline ({oid})")

            order = {
                "order_id": oid,
                "timestamp": now_str,
                "buyer_id": buyer_id,
                "buyer_name": target["name"],
                "item": "Prepaid Voucher Bundle",
                "category": "Digital Goods",
                "amount_dzd": amt,
                "origin": {"name": home_w["name"], "lat": home_w["lat"], "lon": home_w["lon"]},
                "destination": {"name": home_w["name"], "lat": home_w["lat"], "lon": home_w["lon"]},
                "distance_km": 2.4,
                "velocity_kmh": 24.0,
                "attempt_num": step,
                "consecutive_declines": step,
                "is_declined": 1,
                "risk_score": score,
                "verdict": verdict,
                "reason": reason,
                "latency_ms": 29.4,
            }
            STATE["orders"].insert(0, order)
            emitted.append(order)
            await broadcast_event("order", order)
            await asyncio.sleep(0.65)

        return {"status": "ok", "orders": emitted}

    elif scenario == "banned_retry":
        if not STATE["banned_users"]:
            first = generate_transaction(forced_mode="impossible_travel")
            await broadcast_event("order", first)
            await asyncio.sleep(0.4)
        order = generate_transaction(forced_mode="banned_retry")
        await broadcast_event("order", order)
        return {"status": "ok", "orders": [order]}

    raise HTTPException(status_code=400, detail="Unknown simulation scenario")


@app.post("/api/review/{order_id}")
async def analyst_review(order_id: str, decision: str):
    for o in STATE["orders"]:
        if o["order_id"] == order_id:
            if decision == "approve":
                o["verdict"] = "AUTHORIZED"
                o["reason"] = "Cleared by SOC Analyst Override"
                log_audit("ANALYST_APPROVE", f"Approved order {order_id} ({o['amount_dzd']} DZD)", "SOC-ANALYST")
            else:
                o["verdict"] = "FRAUD_BLOCKED"
                o["reason"] = "Confirmed Fraud by SOC Analyst — Account Frozen"
                STATE["banned_users"][o["buyer_id"]] = {
                    "buyer_id": o["buyer_id"],
                    "buyer_name": o["buyer_name"],
                    "reason": o["reason"],
                    "banned_at": datetime.now().strftime("%H:%M:%S"),
                    "home_wilaya": o["origin"]["name"],
                    "home_wilaya_obj": o["origin"],
                    "trigger_order": order_id,
                }
                log_audit("ANALYST_BLOCK", f"Blocked {order_id} and froze {o['buyer_id']}", "SOC-ANALYST")
            await broadcast_event("review_update", o)
            return {"status": "ok", "order": o}
    raise HTTPException(status_code=404, detail="Order not found")


@app.post("/api/unban/{buyer_id}")
async def unban_account(buyer_id: str):
    removed = STATE["banned_users"].pop(buyer_id, None)
    STATE["decline_streaks"][buyer_id] = 0
    log_audit("ACCOUNT_UNBANNED", f"Restored account {buyer_id} to active standing", "SOC-ANALYST")
    await broadcast_event("unban", {"buyer_id": buyer_id})
    return {"status": "ok", "unbanned": buyer_id, "previous": removed}


@app.get("/api/audit")
def get_audit_log():
    return {"audit_log": STATE["audit_log"]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)