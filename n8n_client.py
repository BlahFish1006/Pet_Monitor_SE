"""Minimal n8n webhook client for the Pet Edge Tracking System.

When an event fires (abnormal barking, pet detected, ...), the detectors POST a
small JSON payload to an n8n **Webhook** node. n8n then represents the system's
"Action Output" stage as a live workflow — routing by scenario and performing the
action (notify owner, write log, etc.).

Design note: a webhook failure must never interrupt detection, so all errors are
logged to stderr and swallowed; send_event returns True/False instead of raising.
"""

from __future__ import annotations

import sys

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def build_payload(event_type: str, scenario: int, timestamp: str,
                  source: str, message: str = "", data: dict | None = None) -> dict:
    """Standard event schema posted to n8n."""
    return {
        "event_type": event_type,   # e.g. "abnormal_barking", "pet_detected"
        "scenario": scenario,       # Pet Edge scenario number (0/1/2/3)
        "timestamp": timestamp,
        "source": source,           # "bark_detector" | "dashboard"
        "message": message,
        "data": data or {},
    }


def build_alert_trigger(event_type: str, confidence_pct, timestamp: str,
                        scenario: int | None = None, message: str = "") -> dict:
    """ICD-COMP-UI-001 ALERT_TRIGGER payload (interface B.COMP -> B.UI).

    Carries exactly the three fields the ICD defines — Event_Type, Confidence_%,
    Timestamp — plus optional context for the B.UI action nodes.
    """
    return {
        "interface": "ICD-COMP-UI-001",
        "signal": "ALERT_TRIGGER",
        "source_block": "B.COMP",
        "Event_Type": event_type,           # 越界 danger_zone / 吠叫 abnormal_barking
        "Confidence_%": confidence_pct,      # vision: YOLO box conf ×100; audio: loud_ratio ×100
        "Timestamp": timestamp,
        "scenario": scenario,
        "message": message,
    }


def send_event(webhook_url: str | None, payload: dict, timeout: float = 3.0) -> bool:
    """POST `payload` as JSON to `webhook_url`. Returns True on HTTP 2xx.

    No-op (returns False) when url is empty or requests is unavailable.
    """
    if not webhook_url:
        return False
    if not _HAS_REQUESTS:
        print("[n8n] requests not installed; cannot POST event", file=sys.stderr)
        return False
    try:
        resp = requests.post(webhook_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return True
    except Exception as e:  # connection refused, timeout, non-2xx, ...
        print(f"[n8n] webhook POST failed: {e}", file=sys.stderr)
        return False
