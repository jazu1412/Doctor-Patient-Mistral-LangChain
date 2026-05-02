"""
Demo-only emergency vitals WebSocket: simulated pulse, SpO2, BP, RR, temperature.
Not for clinical use.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["emergency-demo"])


class _SimState:
    def __init__(self) -> None:
        self.hr = 72.0
        self.spo2 = 98.0
        self.bp_sys = 118.0
        self.bp_dia = 78.0
        self.rr = 16.0
        self.temp_f = 98.6

    def step(self) -> dict[str, Any]:
        # Occasional synthetic emergencies for demo (not physiological modeling).
        if random.random() < 0.02:
            self.hr = random.uniform(135, 165)
        elif random.random() < 0.05:
            self.hr = random.uniform(105, 125)
        else:
            self.hr = max(48, min(175, self.hr + random.uniform(-4, 4)))

        if random.random() < 0.015:
            self.spo2 = random.uniform(86, 91)
        else:
            self.spo2 = max(88, min(100, self.spo2 + random.uniform(-0.6, 0.4)))

        self.bp_sys = max(85, min(200, self.bp_sys + random.uniform(-3, 3)))
        self.bp_dia = max(50, min(120, self.bp_dia + random.uniform(-2, 2)))
        self.rr = max(8, min(32, self.rr + random.uniform(-1.2, 1.2)))
        self.temp_f = max(96.5, min(104.5, self.temp_f + random.uniform(-0.08, 0.08)))

        vitals = {
            "heart_rate_bpm": round(self.hr, 1),
            "spo2_pct": round(self.spo2, 1),
            "blood_pressure": {
                "systolic": int(round(self.bp_sys)),
                "diastolic": int(round(self.bp_dia)),
            },
            "respiratory_rate": round(self.rr, 1),
            "temperature_f": round(self.temp_f, 1),
        }
        urgency, alert, msg = _classify(vitals)
        return {
            "ts": int(time.time() * 1000),
            "vitals": vitals,
            "urgency": urgency,
            "alert": alert,
            "message": msg,
        }


def _classify(v: dict[str, Any]) -> tuple[str, bool, str | None]:
    hr = float(v["heart_rate_bpm"])
    spo2 = float(v["spo2_pct"])
    sys_bp = int(v["blood_pressure"]["systolic"])
    temp = float(v["temperature_f"])

    if hr >= 130 or spo2 < 92 or sys_bp >= 180 or temp >= 103.0:
        return (
            "critical",
            True,
            "Critical threshold exceeded — escalate per protocol.",
        )
    if hr >= 100 or spo2 < 95 or sys_bp >= 160 or temp >= 101.0:
        return ("high", True, "Elevated readings — increase monitoring.")
    return ("normal", False, None)


@router.websocket("/emergency/vitals/ws")
async def vitals_demo_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    from app.orchestration.hooks import hook_emergency

    hook_emergency({"channel": "websocket", "path": "/emergency/vitals/ws"})
    state = _SimState()
    try:
        while True:
            payload = state.step()
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.85)
    except WebSocketDisconnect:
        return
