# SentinelAI — Real-Time E-Commerce Fraud & Velocity Defense

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)
![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900?style=flat-square&logo=leaflet)
![Coverage](https://img.shields.io/badge/Coverage-58%20Algerian%20Wilayas-emerald?style=flat-square)

SentinelAI is an automated Security Operations Center (SOC) dashboard and real-time inference engine designed to detect card-testing bursts, account takeovers, and impossible-travel anomalies across Algerian digital commerce platforms.

Built for the **Thirduni 2026 Demo Day (U.S. Embassy Algiers)**, SentinelAI couples a deep neural network with spherical trigonometry kinematics to evaluate incoming checkout payloads under 40 milliseconds.

---

## Architecture Overview

```text
Incoming Checkout Payload (DZD, Coordinates, Merchant Category, Card Status)
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │    Gateway Verification       │
                 │   - Blacklist / Frozen Check  │
                 └───────────────┬───────────────┘
                                 │ Passed
                                 ▼
                 ┌───────────────────────────────┐
                 │  Kinematic Feature Extractor  │
                 │   - Haversine Geodesic Dist   │
                 │   - Velocity (km/h)           │
                 │   - Consecutive Declines      │
                 └───────────────┬───────────────┘
                                 │ Normalized Vector
                                 ▼
                 ┌───────────────────────────────┐
                 │  TensorFlow / Keras Model     │
                 │   - Deep Dense Neural Net     │
                 │   - P(Fraud) Sigmoid Output   │
                 └───────────────┬───────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
  P(Fraud) >= 0.50                                P(Fraud) < 0.50
┌─────────────────────────────┐                 ┌─────────────────────────────┐
│ 🚨 Action: Block & Freeze   │                 │ ✓ Action: Authorized        │
│ - Broadcast SSE "order"     │                 │ - Broadcast SSE "order"     │
│ - Render Bezier Attack Arc  │                 │ - Append to Live Log        │
│ - Auto-ban User Profile     │                 │ - Update DZD Revenue KPIs   │
└─────────────────────────────┘                 └─────────────────────────────┘
```

---

## Mathematical & Kinematic Formulations

### 1. Great-Circle Geodesic Distance
To prevent attackers using foreign exit nodes (VPNs/proxies) from evading local velocity caps, travel distance is computed using the spherical Earth radius ($R \approx 6,371\text{ km}$) and the Haversine formula:

$$\Delta\sigma = 2 \arcsin \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}$$

$$d = R \cdot \Delta\sigma$$

### 2. Physical Velocity Formulation
Velocity evaluates whether consecutive transactions exceed physical human travel speeds ($v > 900\text{ km/h}$ flag):

$$v = \frac{d}{\Delta t}$$

### 3. Neural Classification
Vectors pass through a standard scalar transform $z = \frac{x - \mu}{\sigma}$ across trained feature dimensions (Amount, Velocity, Attempt Count, Decline Flag, Lat/Lon, One-Hot Categories) and are classified by a dense neural network:

$$P(\text{Fraud}) = \sigma\left(\mathbf{W}_3 \cdot \text{ReLU}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + b_3\right)$$

---

## Core Capabilities

- **58 Algerian Wilayas Monitored:** Geocoded coordinates, regional IP blocks, and localized customer pools spanning all 58 provinces (01 - Adrar to 58 - El Meniaa).
- **Sequential Card-Testing Interception:** Live dashboard traces every failed card attempt (`Declined 1/3`, `Declined 2/3`) in real time before executing automated account freezing upon the 3rd attempt.
- **Offline-First Resilience:** Zero external CDN dependencies at runtime. Uses local vector Natural Earth 110m GeoJSON styling, pre-compiled CSS, and embedded WebAudio synthesizers to operate without internet on venue Wi-Fi.
- **Dynamic Bézier Threat Arcs:** When impossible travel triggers, the map dynamically pans (`flyToBounds`) and animates a quadratic Bézier curve connecting the customer's Algerian home Wilaya to the foreign exit node with full velocity metadata.
- **Human-in-the-Loop Analyst Operations:** Dedicated **Needs Review** queue with single-click manual override (`Approve` / `Block`) and real-time state synchronization across all open sessions.
- **Case-File Forensics:** Modal inspection panel with a chronological order timeline, numbered historical order trails on a mini GIS map, and immediate unban restoration.

---

## Project Structure

```text
├── static/
│   ├── css/
│   │   └── leaflet.css              # Cached local Leaflet styles
│   ├── js/
│   │   ├── leaflet.js               # Cached local Leaflet engine
│   │   └── chart.umd.min.js         # Offline chart rendering
│   └── geojson/
│       └── ne_110m_countries.geojson# Offline vector basemap
├── SentinelAI_Algeria_Fraud_Dataset_2500.xlsx # Training corpus
├── train.py                         # Model architecture, training & artifact export
├── server.py                        # FastAPI streaming backend & simulation engine
├── index.html                       # Real-time dark-mode SOC dashboard
├── setup_offline.py                 # Asset pre-fetcher for air-gapped demo runs
├── requirements.txt                 # Runtime dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Modern Web Browser (Google Chrome, Microsoft Edge, or Mozilla Firefox)

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/Zine-Elabidine-Islam/Sentinel-AI-Fraud-Defense.git
cd Sentinel-AI-Fraud-Defense
pip install -r requirements.txt
```

*(Contents of `requirements.txt`: `fastapi`, `uvicorn`, `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `openpyxl`, `joblib`)*

### 3. Pre-Cache Offline Assets (Run once while connected to internet)
Download the local basemaps and libraries into `/static`:

```bash
python setup_offline.py
```

### 4. Train the Model
Fit the neural network and export the production pipeline (`fraud_model.keras`, `scaler.pkl`, `feature_columns.pkl`):

```bash
python train.py
```

### 5. Launch the SentinelAI SOC
Start the local ASGI web server:

```bash
python -m uvicorn server:app --reload --port 8000
```

Open your browser at **`http://localhost:8000`**.

---

## Presenter Controls & Keyboard Shortcuts

SentinelAI features a dedicated hidden control deck designed for live demonstrations:

| Key | Function |
| :---: | :--- |
| **`P`** | Toggle the **Presenter Control Panel** (or triple-click the SentinelAI shield logo) |
| **`F`** | Toggle **Fullscreen Mode** |
| **`M`** | Toggle **WebAudio Acoustic Alarm** (audible warning on intercepted attacks) |
| **`1`** | Switch to **Live Operations Log** tab |
| **`2`** | Switch to **Needs Review** analyst tab |
| **`3`** | Switch to **Banned Accounts Queue** tab |
| **`Esc`** | Close any open modal or audit drawer |

---

## Demo Day Simulation Scenarios

Using the Presenter Panel (**`P`**), you can demonstrate three specific threat scenarios live to an audience:

1. **🚨 Impossible Travel Attack:** Dispatches a high-ticket transaction from a foreign anonymizing exit node (London, Paris, Frankfurt), plotting a cross-Mediterranean velocity arc ($v > 8,000\text{ km/h}$) and auto-freezing the profile.
2. **💳 Sequential Card Decline:** Dispatches a 3-step rapid card-testing sequence inside an Algerian Wilaya, displaying intermediate `Declined (1/3)` warnings before auto-suspending the account on the 3rd attempt.
3. **🛑 Banned Account Retry:** Demonstrates gateway-level policy enforcement when an already-blacklisted user attempts to place a new order.

---

## License & Acknowledgments

Developed by **Lakhlef Zine Elabidine** for the **Thirduni AI Training Program** supported by the **U.S. Embassy in Algiers**.

Distributed under the MIT License.
