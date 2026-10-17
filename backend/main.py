"""
main.py — Servidor FastAPI del monitor de mesas.

Flujo de datos:

    iPhone (camera.html) ──JPEG por WebSocket──▶  /ws/camera
                                                      │
                                            detector.py (personas / platos)
                                                      │
                                            automaton.py (4 estados + alertas)
                                                      │
    PC (dashboard.html)  ◀──JSON + fotograma──────  /ws/dashboard

Ejecutar (desde la carpeta backend/):
    python main.py                      # HTTPS si existe certs/, HTTP si no
    DETECTOR=sim python main.py         # modo demo sin visión real
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import time
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from automaton import AutomatonConfig, TableStateMachine, states_catalog
from detector import create_detector, resize_to_width

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
CERT_DIR = BASE_DIR / "certs"
PORT = int(os.getenv("PORT", "8000"))
PROC_WIDTH = int(os.getenv("PROC_WIDTH", "640"))     # ancho al que se analiza cada fotograma

app = FastAPI(title="Restaurant Table Monitor")

# ------------------------------------------------------------------ estado global
detector = create_detector()                          # heuristic | yolo | sim  (env DETECTOR)
machine = TableStateMachine(AutomatonConfig(
    debounce_seconds=float(os.getenv("DEBOUNCE_SECONDS", "2")),
    dirty_alert_seconds=float(os.getenv("DIRTY_ALERT_SECONDS", "15")),
    waiting_alert_seconds=float(os.getenv("WAITING_ALERT_SECONDS", "20")),
))
dashboards: Set[WebSocket] = set()
_busy = False                                         # ¿hay un fotograma procesándose?
_tasks: Set[asyncio.Task] = set()                     # referencias para que no se recojan
stats = {"frames_in": 0, "frames_processed": 0, "frames_dropped": 0, "fps": 0.0, "last_frame_at": None}
_last_proc_t: Optional[float] = None


# ----------------------------------------------------------------------- helpers
def get_lan_ips() -> list[str]:
    ips: list[str] = []
    try:  # truco UDP: no envía paquetes, sólo averigua qué interfaz saldría a la red
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def _analyze(jpeg: bytes):
    """CPU-bound: decodifica, detecta y dibuja. Se ejecuta en un hilo aparte."""
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    frame = resize_to_width(frame, PROC_WIDTH)
    det = detector.detect(frame)
    vis = detector.draw(frame, det)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return (det, buf.tobytes()) if ok else None


async def broadcast(payload: dict) -> None:
    if not dashboards:
        return
    text = json.dumps(payload)
    clients = list(dashboards)
    results = await asyncio.gather(*(c.send_text(text) for c in clients), return_exceptions=True)
    for c, r in zip(clients, results):
        if isinstance(r, Exception):
            dashboards.discard(c)


async def handle_frame(jpeg: bytes) -> bool:
    """Procesa un fotograma. Devuelve False si se descartó (sistema ocupado)."""
    global _busy, _last_proc_t
    stats["frames_in"] += 1
    if _busy:                                   # backpressure: descartar en vez de acumular latencia
        stats["frames_dropped"] += 1
        return False
    _busy = True
    try:
        out = await asyncio.to_thread(_analyze, jpeg)
        if out is None:
            return False
        det, vis_jpeg = out
        events = machine.update(det.personas, det.platos)

        now = time.monotonic()
        if _last_proc_t is not None and now > _last_proc_t:
            inst = 1.0 / (now - _last_proc_t)
            stats["fps"] = inst if stats["fps"] == 0 else 0.8 * stats["fps"] + 0.2 * inst
        _last_proc_t = now
        stats["frames_processed"] += 1
        stats["last_frame_at"] = time.time()

        await broadcast({
            "type": "frame",
            "frame": base64.b64encode(vis_jpeg).decode("ascii"),
            "detection": det.to_dict(),
            "state": machine.snapshot(),
            "events": events,
            "fps": round(stats["fps"], 1),
        })
        return True
    finally:
        _busy = False


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


# ------------------------------------------------------------------ WebSockets
@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    """Recibe fotogramas JPEG (mensajes binarios) desde el iPhone."""
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                # No se hace 'await' aquí: así el bucle sigue leyendo y los fotogramas
                # que lleguen con el sistema ocupado se descartan (latencia baja).
                _spawn(handle_frame(msg["bytes"]))
            elif msg.get("text"):
                try:
                    if json.loads(msg["text"]).get("type") == "ping":
                        await ws.send_text('{"type":"pong"}')
                except ValueError:
                    pass
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    """Envía al dashboard fotogramas procesados, estado y eventos."""
    await ws.accept()
    dashboards.add(ws)
    try:
        await ws.send_text(json.dumps({
            "type": "init",
            "detector": detector.name,
            "config": vars(machine.config),
            "states": states_catalog(),
            "state": machine.snapshot(),
            "events": machine.recent_events(50),
        }))
        while True:
            await ws.receive_text()             # sólo mantiene viva la conexión
    except WebSocketDisconnect:
        pass
    finally:
        dashboards.discard(ws)


# ---------------------------------------------------------- transporte alternativo
@app.post("/api/frame")
async def post_frame(request: Request):
    """
    Alternativa HTTP al WebSocket (camera.html la usa sola si el WebSocket seguro
    falla, algo posible en iOS con certificados autofirmados).
    """
    body = await request.body()
    if not body:
        raise HTTPException(400, "Cuerpo vacío")
    _spawn(handle_frame(body))
    return {"ok": True}


# -------------------------------------------------------------------- REST API
class ConfigUpdate(BaseModel):
    debounce_seconds: Optional[float] = Field(None, ge=0)
    dirty_alert_seconds: Optional[float] = Field(None, gt=0)
    waiting_alert_seconds: Optional[float] = Field(None, gt=0)


@app.get("/api/health")
async def health():
    last = stats["last_frame_at"]
    return {"ok": True, "detector": detector.name, "dashboards": len(dashboards),
            "last_frame_age_s": None if last is None else round(time.time() - last, 1), **{
                k: stats[k] for k in ("frames_in", "frames_processed", "frames_dropped")}}


@app.get("/api/state")
async def get_state():
    return machine.snapshot()


@app.get("/api/events")
async def get_events(n: int = 50):
    return machine.recent_events(max(1, min(n, 200)))


@app.get("/api/config")
async def get_config():
    return vars(machine.config)


@app.post("/api/config")
async def set_config(update: ConfigUpdate):
    try:
        cfg = machine.configure(**update.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(400, str(e))
    await broadcast({"type": "config", "config": cfg})
    return cfg


@app.post("/api/reset")
async def reset():
    machine.reset()
    detector.reset()
    await broadcast({"type": "reset", "state": machine.snapshot(), "events": []})
    return {"ok": True}


# ------------------------------------------------------------------- páginas
def _page(name: str) -> FileResponse:
    path = FRONTEND_DIR / name
    if not path.exists():
        raise HTTPException(404, f"No se encontró {path}")
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/camera")
@app.get("/camera.html")
async def camera_page():
    return _page("camera.html")


@app.get("/dashboard")
@app.get("/dashboard.html")
async def dashboard_page():
    return _page("dashboard.html")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    scheme = request.url.scheme
    port = request.url.port or (443 if scheme == "https" else 80)
    links = "".join(
        f'<li>iPhone: <a href="{scheme}://{ip}:{port}/camera">{scheme}://{ip}:{port}/camera</a></li>'
        for ip in get_lan_ips()) or "<li>No se detectó IP de red local</li>"
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monitor de mesas</title>
<body style="font-family:system-ui;background:#0f1419;color:#e6edf3;max-width:640px;margin:40px auto;padding:0 16px">
<h1>Monitor de mesas</h1>
<ul style="line-height:2"><li><a style="color:#58a6ff" href="/dashboard">Dashboard (PC)</a></li>
<li><a style="color:#58a6ff" href="/camera">Cámara (esta máquina, para probar con webcam)</a></li>{links}</ul>
<p>Detector activo: <b>{detector.name}</b> · <a style="color:#58a6ff" href="/api/health">/api/health</a> · <a style="color:#58a6ff" href="/docs">/docs</a></p></body>"""


# -------------------------------------------------------------------------- main
if __name__ == "__main__":
    import uvicorn

    cert, key = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"
    use_tls = cert.exists() and key.exists()
    scheme = "https" if use_tls else "http"
    ips = get_lan_ips()

    print("\n" + "=" * 64)
    print(f"  Detector: {detector.name}   |   Protocolo: {scheme.upper()}   |   Puerto: {PORT}")
    print(f"  Dashboard (PC):  {scheme}://localhost:{PORT}/dashboard")
    for ip in ips:
        print(f"  Cámara (iPhone): {scheme}://{ip}:{PORT}/camera")
    if not use_tls:
        print("\n  ⚠ Sin HTTPS: el iPhone NO dejará usar la cámara. Genera el certificado:")
        print("      python make_cert.py   y vuelve a ejecutar  python main.py")
        print("    (en esta misma PC, http://localhost sí funciona para probar con la webcam)")
    print("=" * 64 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info",
                ssl_keyfile=str(key) if use_tls else None,
                ssl_certfile=str(cert) if use_tls else None)
