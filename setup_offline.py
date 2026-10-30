"""
Downloads Leaflet, Chart.js, and Natural Earth 110m GeoJSON into /static
so SentinelAI runs 100% offline with Wi-Fi disabled on Demo Day.
"""
import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

DIRS = [
    STATIC_DIR / "css",
    STATIC_DIR / "js",
    STATIC_DIR / "geojson",
]

ASSETS = {
    STATIC_DIR / "css" / "leaflet.css": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
    STATIC_DIR / "js" / "leaflet.js": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
    STATIC_DIR / "js" / "chart.umd.min.js": "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js",
    STATIC_DIR / "geojson" / "ne_110m_countries.geojson": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
}

def main():
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)

    for dest, url in ASSETS.items():
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[✓] Already cached: {dest.relative_to(BASE_DIR)}")
            continue
        try:
            print(f"[*] Downloading {url} -> {dest.relative_to(BASE_DIR)} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                dest.write_bytes(resp.read())
            print(f"[+] Saved ({dest.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"[!] Could not download {url}: {e}")
            print("    (Built-in fallback in index.html will still work offline.)")

    print("\n[✓] Offline bundle ready in ./static!")

if __name__ == "__main__":
    main()