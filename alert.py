#!/usr/bin/env python3
"""
Adaptive Alert Triage — Alert Flood Generator (FIXED)

Sends realistic alerts to the Docker server's /ingest/alerts endpoint.
The server queues them so the RL environment uses real alert distributions
instead of purely synthetic ones.

Fixes from original:
  1. Field name was "type" — server expects "alert_type"
  2. Added --burst mode to send correlated chains (tests hard task)
  3. Server health check before flooding
  4. Correlation group metadata included

Usage:
    python alert.py --count 500
    python alert.py --count 200 --burst --rate 0.02
    python alert.py --server https://your-ngrok-url.ngrok.io --count 1000
"""

import requests
import random
import time
import json
import argparse
import threading
from typing import List, Dict

ALERT_TYPES = ["CPU", "MEMORY", "DISK", "NETWORK", "APPLICATION", "SECURITY"]

# Realistic correlation chains (mimics utils.generate_correlated_alerts)
CORRELATION_CHAINS = [
    ["CPU", "MEMORY", "APPLICATION"],
    ["NETWORK", "APPLICATION", "APPLICATION"],
    ["DISK", "MEMORY", "APPLICATION"],
    ["SECURITY", "NETWORK", "APPLICATION"],
]


def generate_alert(alert_id: int, force_type: str = None) -> Dict:
    """
    Generate a single realistic alert.

    FIX: field is "alert_type" not "type" — must match the server's
    AlertIngestionRequest schema.
    """
    severity   = round(random.uniform(0.25, 0.99), 2)
    confidence = round(random.uniform(0.45, 0.99), 2)

    # Higher-severity alerts tend to have higher detection confidence
    if severity > 0.85:
        confidence = round(random.uniform(0.80, 0.99), 2)
    elif severity < 0.40:
        confidence = round(random.uniform(0.40, 0.65), 2)

    atype = force_type if force_type else random.choice(ALERT_TYPES)

    return {
        "id":               f"prod-{alert_id:08d}",
        "visible_severity": severity,
        "confidence":       confidence,
        "type":             atype,
        "age":              0,
    }


def generate_correlated_burst(start_id: int) -> List[Dict]:
    """Generate a set of 3 correlated alerts (simulates cascading failure)."""
    chain = random.choice(CORRELATION_CHAINS)
    alerts = []
    for i, atype in enumerate(chain):
        base_sev = round(0.60 + i * 0.12 + random.uniform(-0.05, 0.05), 2)
        base_sev = min(base_sev, 0.99)
        alert = generate_alert(start_id + i, force_type=atype)
        alert["visible_severity"] = base_sev
        alert["confidence"]       = round(random.uniform(0.75, 0.95), 2)
        alert["is_correlated"]    = True
        alert["correlation_chain"] = chain
        alerts.append(alert)
    return alerts


def check_server(server_url: str) -> bool:
    try:
        r = requests.get(f"{server_url}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def send_alerts(server_url: str, alerts: List[Dict], rate: float) -> int:
    session = requests.Session()
    queued  = 0
    print(f"\n🚨  Sending {len(alerts)} alerts to {server_url}/ingest/alerts ...")

    for i, alert in enumerate(alerts):
        try:
            resp = session.post(
                f"{server_url}/ingest/alerts",
                json=alert,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if resp.status_code == 200:
                queued += 1
                if i % 100 == 0 and i > 0:
                    q = resp.json().get("queued", "?")
                    print(f"  {i+1:>5}/{len(alerts)}  queued={q}")
            else:
                print(f"  ❌  alert {i}: HTTP {resp.status_code} — {resp.text[:80]}")

        except Exception as e:
            print(f"  ⚠   alert {i}: {e}")

        time.sleep(rate)

    print(f"\n✅  Flood done — {queued}/{len(alerts)} accepted")
    return queued


def monitor_loop(server_url: str):
    """Background thread: print live metrics every 5 s."""
    while True:
        time.sleep(5)
        try:
            d = requests.get(f"{server_url}/metrics", timeout=3).json()
            print(
                f"\n📈  METRICS  score={d.get('mean_score', '?')}  "
                f"queue={d.get('queue_size', '?')}  "
                f"active={d.get('active_alerts', '?')}  "
                f"episodes={d.get('episodes_completed', '?')}"
            )
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Flood the alert triage server")
    parser.add_argument("--server",  default="https://731a-2401-4900-619d-77b-a572-ef53-21cd-5507.ngrok-free.app")
    parser.add_argument("--count",   type=int,   default=500)
    parser.add_argument("--rate",    type=float, default=0.05,
                        help="Seconds between alerts (default 0.05)")
    parser.add_argument("--burst",   action="store_true",
                        help="Interleave correlated 3-alert bursts (hard task training)")
    parser.add_argument("--monitor", action="store_true",
                        help="Print live metrics while flooding")
    args = parser.parse_args()

    print("=" * 55)
    print("  Alert Triage — Alert Flood Generator")
    print("=" * 55)
    print(f"  Server : {args.server}")
    print(f"  Count  : {args.count}")
    print(f"  Rate   : {args.rate}s / alert")
    print(f"  Burst  : {args.burst}")

    # Sanity check
    if not check_server(args.server):
        print(f"\n❌  Server not reachable at {args.server}")
        print("    Start with:  docker compose up --build -d")
        return

    print("  Server : ✅ reachable\n")

    # Build alert list
    alerts: List[Dict] = []
    i = 0
    while len(alerts) < args.count:
        if args.burst and random.random() < 0.15:      # 15% chance of a burst
            burst = generate_correlated_burst(i)
            alerts.extend(burst)
            i += len(burst)
        else:
            alerts.append(generate_alert(i))
            i += 1

    alerts = alerts[:args.count]

    # Show sample
    print("Sample alerts:")
    for a in alerts[:2]:
        print(f"  {json.dumps(a)}")
    print()

    # Start monitor if requested
    if args.monitor:
        t = threading.Thread(target=monitor_loop, args=(args.server,), daemon=True)
        t.start()

    # Flood
    queued = send_alerts(args.server, alerts, args.rate)

    # Final metrics
    print("\nFinal server metrics:")
    try:
        print(json.dumps(
            requests.get(f"{args.server}/metrics", timeout=5).json(),
            indent=2
        ))
    except Exception as e:
        print(f"  (could not fetch: {e})")


if __name__ == "__main__":
    main()