"""
automaton.py — Máquina de estados finitos para una mesa de restaurante.

Entrada (lo que reporta el detector en cada fotograma):
    personas : bool                                   ¿hay gente en la mesa?
    platos   : "vacios" | "con_comida" | "ninguno"    estado de los platos

Los 4 estados lógicos:

    ┌────┬──────────────────────┬──────────┬────────────────────────────────┐
    │ ID │ Estado               │ Personas │ Platos                         │
    ├────┼──────────────────────┼──────────┼────────────────────────────────┤
    │ 1  │ VACIA                │ no       │ ninguno         → sin alerta   │
    │ 2  │ CON_RESTOS           │ no       │ vacios/comida   → ALERTA limpieza
    │ 3  │ SENTADOS_SIN_COMIDA  │ sí       │ ninguno         → ALERTA atención (si pasa X s)
    │ 4  │ COMIENDO             │ sí       │ vacios/comida   → sin alerta   │
    └────┴──────────────────────┴──────────┴────────────────────────────────┘

Reglas temporales:
  * DEBOUNCE: un estado nuevo debe observarse de forma continua durante
    `debounce_seconds` antes de aceptarlo (evita parpadeos por fallos puntuales
    del detector). El cronómetro del estado arranca cuando la condición
    *empezó* a observarse, no cuando terminó el debounce.
  * ALERTA DE LIMPIEZA:  estado 2 persistiendo >= `dirty_alert_seconds`.
  * ALERTA DE ATENCIÓN:  estado 3 persistiendo >= `waiting_alert_seconds`.
  * Cada alerta se dispara UNA vez por permanencia y se resuelve sola cuando
    el estado cambia.
  * Si pasan más de `stale_after_seconds` sin fotogramas (cámara caída), se
    reinician los cronómetros al volver la señal, para no disparar alertas
    falsas basadas en datos viejos.

La clase no depende de OpenCV ni de la red: es lógica pura y se puede probar
inyectando el parámetro `now` (ver test_automaton.py).
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Deque, Dict, List, Optional

PLATOS_VALIDOS = ("vacios", "con_comida", "ninguno")


class TableState(str, Enum):
    VACIA = "vacia"                              # Estado 1
    CON_RESTOS = "con_restos"                    # Estado 2
    SENTADOS_SIN_COMIDA = "sentados_sin_comida"  # Estado 3
    COMIENDO = "comiendo"                        # Estado 4


# Metadatos de cada estado (el dashboard los usa para pintar la leyenda).
STATE_META: Dict[TableState, dict] = {
    TableState.VACIA: dict(
        id=1, label="Mesa vacía", severity="ok",
        description="Sin personas ni objetos. Estado natural."),
    TableState.CON_RESTOS: dict(
        id=2, label="Mesa con restos", severity="warn",
        description="Sin personas pero con platos. Requiere limpieza."),
    TableState.SENTADOS_SIN_COMIDA: dict(
        id=3, label="Clientes sin comida", severity="warn",
        description="Personas sentadas sin comida servida. Requiere atención."),
    TableState.COMIENDO: dict(
        id=4, label="Clientes comiendo", severity="ok",
        description="Personas sentadas con platos. Disfrutando."),
}


def states_catalog() -> List[dict]:
    """Lista de estados con sus metadatos (para enviar al frontend)."""
    return [dict(key=s.value, **STATE_META[s]) for s in TableState]


def classify(personas: bool, platos: str) -> TableState:
    """Tabla de verdad (personas, platos) -> estado lógico."""
    hay_platos = platos in ("vacios", "con_comida")
    if personas:
        return TableState.COMIENDO if hay_platos else TableState.SENTADOS_SIN_COMIDA
    return TableState.CON_RESTOS if hay_platos else TableState.VACIA


@dataclass
class AutomatonConfig:
    debounce_seconds: float = 2.0        # estabilidad requerida para cambiar de estado
    dirty_alert_seconds: float = 15.0    # estado 2 -> alerta de limpieza   (demo; en producción ~120-300)
    waiting_alert_seconds: float = 20.0  # estado 3 -> alerta de atención   (demo; en producción ~180-600)
    stale_after_seconds: float = 10.0    # hueco sin fotogramas que invalida cronómetros


def _fmt(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class TableStateMachine:
    def __init__(self, config: Optional[AutomatonConfig] = None):
        self.config = config or AutomatonConfig()
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self.history: Deque[dict] = deque(maxlen=200)
        self._reset_internal()

    # ------------------------------------------------------------------ API
    def update(self, personas: bool, platos: str, now: Optional[float] = None) -> List[dict]:
        """
        Procesa UNA observación del detector. Devuelve la lista de eventos
        nuevos (cambios de estado, alertas, alertas resueltas).
        `now` (segundos, reloj monotónico) se puede inyectar para pruebas.
        """
        if platos not in PLATOS_VALIDOS:
            raise ValueError(f"platos debe ser uno de {PLATOS_VALIDOS}, recibido: {platos!r}")
        now = time.monotonic() if now is None else float(now)

        with self._lock:
            events: List[dict] = []
            observed = classify(bool(personas), platos)
            self._last_obs = {"personas": bool(personas), "platos": platos}

            stale = (self._last_update is not None
                     and now - self._last_update > self.config.stale_after_seconds)
            self._last_update = now

            # Primera observación, o regreso tras pérdida de señal: re-sincronizar.
            if self.state_since is None or stale:
                self._resync(observed, now, events, resumed=stale)
                return events

            if observed == self.state:
                self._pending = None
                self._pending_since = None
            else:
                if self._pending != observed:       # candidato nuevo: arranca debounce
                    self._pending = observed
                    self._pending_since = now
                if now - self._pending_since >= self.config.debounce_seconds:
                    self._transition(observed, self._pending_since, now, events)

            self._check_alerts(now, events)
            return events

    def snapshot(self, now: Optional[float] = None) -> dict:
        """Foto del estado actual, lista para serializar a JSON."""
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            meta = STATE_META[self.state]
            elapsed = 0.0 if self.state_since is None else max(0.0, now - self.state_since)
            threshold = self._threshold_for(self.state)
            return {
                "state": self.state.value,
                "state_id": meta["id"],
                "label": meta["label"],
                "description": meta["description"],
                "severity": meta["severity"],
                "seconds_in_state": round(elapsed, 1),
                "pending_state": self._pending.value if self._pending else None,
                "alert": dict(self._active_alert) if self._active_alert else None,
                "alert_threshold": threshold,
                "seconds_to_alert": (None if threshold is None or self._active_alert
                                     else round(max(0.0, threshold - elapsed), 1)),
                "observation": dict(self._last_obs),
            }

    def configure(self, **kwargs) -> dict:
        """Actualiza umbrales en caliente. Devuelve la configuración resultante."""
        with self._lock:
            for key, value in kwargs.items():
                if value is None:
                    continue
                if not hasattr(self.config, key):
                    raise ValueError(f"Parámetro desconocido: {key}")
                value = float(value)
                if value < 0 or (key.endswith("_alert_seconds") and value == 0):
                    raise ValueError(f"Valor inválido para {key}: {value}")
                setattr(self.config, key, value)
            return asdict(self.config)

    def reset(self) -> None:
        with self._lock:
            self.history.clear()
            self._reset_internal()

    def recent_events(self, n: int = 50) -> List[dict]:
        with self._lock:
            return list(self.history)[-n:]

    # ------------------------------------------------------------ internals
    def _reset_internal(self) -> None:
        self.state = TableState.VACIA
        self.state_since: Optional[float] = None
        self._pending: Optional[TableState] = None
        self._pending_since: Optional[float] = None
        self._last_update: Optional[float] = None
        self._active_alert: Optional[dict] = None
        self._last_obs = {"personas": False, "platos": "ninguno"}

    def _threshold_for(self, state: TableState) -> Optional[float]:
        if state is TableState.CON_RESTOS:
            return self.config.dirty_alert_seconds
        if state is TableState.SENTADOS_SIN_COMIDA:
            return self.config.waiting_alert_seconds
        return None

    def _emit(self, events: List[dict], type_: str, severity: str, message: str,
              kind: Optional[str] = None) -> dict:
        ev = {
            "id": next(self._ids),
            "ts": _now_iso(),
            "type": type_,          # state_change | alert | alert_resolved | info
            "kind": kind,           # limpieza | atencion | None
            "severity": severity,   # info | warning
            "state": self.state.value,
            "message": message,
        }
        self.history.append(ev)
        events.append(ev)
        return ev

    def _resolve_alert(self, events: List[dict], reason: str) -> None:
        if self._active_alert:
            kind = self._active_alert["kind"]
            nombre = {"limpieza": "limpieza", "atencion": "atención"}[kind]
            self._emit(events, "alert_resolved", "info",
                       f"Alerta de {nombre} resuelta: {reason}", kind=kind)
            self._active_alert = None

    def _resync(self, observed: TableState, now: float, events: List[dict], resumed: bool) -> None:
        self._resolve_alert(events, "reinicio de cronómetros")
        self.state = observed
        self.state_since = now
        self._pending = self._pending_since = None
        msg = ("Señal de cámara recuperada; cronómetros reiniciados. Estado: "
               if resumed else "Monitoreo iniciado. Estado inicial: ")
        self._emit(events, "info", "info", msg + STATE_META[observed]["label"])

    def _transition(self, new_state: TableState, started_at: float, now: float,
                    events: List[dict]) -> None:
        prev = self.state
        self._resolve_alert(events, "cambió el estado de la mesa")
        self.state = new_state
        self.state_since = started_at        # el cronómetro cuenta desde que empezó la condición
        self._pending = self._pending_since = None
        self._emit(events, "state_change", "info",
                   f"{STATE_META[prev]['label']} → {STATE_META[new_state]['label']}")

    def _check_alerts(self, now: float, events: List[dict]) -> None:
        if self._active_alert is not None:
            return
        threshold = self._threshold_for(self.state)
        if threshold is None:
            return
        elapsed = now - self.state_since
        if elapsed < threshold:
            return
        if self.state is TableState.CON_RESTOS:
            kind = "limpieza"
            msg = f"Mesa sin clientes con platos desde hace {_fmt(elapsed)}. Requiere limpieza."
        else:
            kind = "atencion"
            msg = f"Clientes sentados sin comida servida desde hace {_fmt(elapsed)}. Requiere atención."
        ev = self._emit(events, "alert", "warning", msg, kind=kind)
        self._active_alert = {"kind": kind, "message": msg, "raised_at": ev["ts"], "event_id": ev["id"]}
