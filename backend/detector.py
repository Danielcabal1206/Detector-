"""
detector.py — Visión por computador para la mesa.

Contrato: todos los detectores reciben un fotograma BGR (numpy, ya redimensionado)
y devuelven un objeto `Detection`:

    personas : bool                                  ¿hay personas en la mesa?
    platos   : "vacios" | "con_comida" | "ninguno"   estado de los platos

Hay TRES implementaciones intercambiables (variable de entorno DETECTOR):

  1. heuristic (por defecto) — OpenCV clásico, sin descargas ni GPU:
       · Personas: cascadas Haar (rostro + torso) y, opcionalmente, HOG de peatones.
       · Platos:   círculos (Hough) que contrastan con la mesa y tienen un borde
                   uniforme; si el centro se aleja del color del borde -> hay comida.
     Es un PROTOTIPO: funciona bien con luz decente, platos redondos claros sobre
     mantel de otro color y gente mirando hacia la cámara. Con platos de colores,
     sombras fuertes o gente de espaldas, se equivoca. Para producción, usa YOLO.

  2. yolo — Ultralytics YOLO (pip install ultralytics). Ver clase YoloDetector.
       DETECTOR=yolo  YOLO_MODEL=yolov8n.pt   (COCO, funciona "out of the box")
       DETECTOR=yolo  YOLO_MODEL=mi_modelo.pt (recomendado: modelo propio)

  3. sim — Simulador: recorre los 4 estados cada SIM_PERIOD segundos sin mirar la
     imagen. Sirve para probar el dashboard y las alertas sin escenario real.

Región de interés (opcional): TABLE_ROI="x,y,w,h" con valores normalizados 0-1,
p. ej. TABLE_ROI="0.2,0.3,0.6,0.5". Sólo cuentan los objetos cuyo centro cae dentro
(así ignoras al personal que pasa por el pasillo).

Parámetros de las heurísticas: constantes al inicio de HeuristicDetector.
"""
from __future__ import annotations

import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h

PLATOS_VACIOS = "vacios"
PLATOS_CON_COMIDA = "con_comida"
PLATOS_NINGUNO = "ninguno"


# --------------------------------------------------------------------------- tipos
@dataclass
class PlateItem:
    box: Box
    estado: str          # "vacios" | "con_comida"
    score: float = 0.0   # heurística: proporción de "comida" en el plato


@dataclass
class Detection:
    personas: bool
    platos: str
    personas_boxes: List[Box] = field(default_factory=list)
    platos_items: List[PlateItem] = field(default_factory=list)
    raw_personas: bool = False          # antes del suavizado temporal
    raw_platos: str = PLATOS_NINGUNO    # antes del suavizado temporal

    def to_dict(self) -> dict:
        return {
            "personas": self.personas,
            "platos": self.platos,
            "n_personas": len(self.personas_boxes),
            "n_platos": len(self.platos_items),
        }


# ------------------------------------------------------------------------ utilidades
def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def parse_roi(value: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """'x,y,w,h' normalizados -> tupla, o None si no es válido."""
    if not value:
        return None
    try:
        x, y, w, h = [float(v) for v in value.split(",")]
    except ValueError:
        return None
    if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0):
        return None
    return (x, y, min(w, 1 - x), min(h, 1 - y))


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _merge_boxes(boxes: List[Box], thr: float = 0.35) -> List[Box]:
    """Supresión simple de cajas solapadas (conserva la más grande)."""
    kept: List[Box] = []
    for b in sorted(boxes, key=lambda r: r[2] * r[3], reverse=True):
        if all(_iou(b, k) < thr for k in kept):
            kept.append(b)
    return kept


# ------------------------------------------------------------------------ base común
class BaseDetector:
    name = "base"

    def __init__(self, roi: Optional[str] = None, smooth_window: int = 5):
        self.roi = parse_roi(roi if roi is not None else os.getenv("TABLE_ROI"))
        self._people_hist: Deque[bool] = deque(maxlen=smooth_window)
        self._plates_hist: Deque[str] = deque(maxlen=smooth_window)

    # --- interfaz ---
    def detect(self, frame_bgr: np.ndarray) -> Detection:
        raise NotImplementedError

    def reset(self) -> None:
        self._people_hist.clear()
        self._plates_hist.clear()

    # --- ayudas ---
    def _in_roi(self, box: Box, shape) -> bool:
        if self.roi is None:
            return True
        h, w = shape[:2]
        x, y, bw, bh = box
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        rx, ry, rw, rh = self.roi
        return rx <= cx <= rx + rw and ry <= cy <= ry + rh

    def _smooth(self, raw_personas: bool, raw_platos: str) -> Tuple[bool, str]:
        """
        Suavizado temporal (voto sobre los últimos N fotogramas): las cascadas
        Haar fallan de forma intermitente y un solo fotograma no debe cambiar
        el resultado. (El autómata aplica además su propio debounce.)
        """
        self._people_hist.append(bool(raw_personas))
        self._plates_hist.append(raw_platos)
        personas = sum(self._people_hist) >= min(2, len(self._people_hist))
        counts = Counter(self._plates_hist)
        platos = max(counts, key=lambda k: (counts[k], k == raw_platos))
        return personas, platos

    def draw(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        """Devuelve una copia del fotograma con las detecciones dibujadas."""
        out = frame.copy()
        h, w = out.shape[:2]
        if self.roi:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(out, (int(rx * w), int(ry * h)), (int((rx + rw) * w), int((ry + rh) * h)),
                          (0, 215, 255), 1)
        for (x, y, bw, bh) in det.personas_boxes:
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (255, 140, 0), 2)
            cv2.putText(out, "persona", (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 140, 0), 1, cv2.LINE_AA)
        for it in det.platos_items:
            x, y, bw, bh = it.box
            color = (0, 140, 255) if it.estado == PLATOS_CON_COMIDA else (60, 200, 60)
            cv2.circle(out, (x + bw // 2, y + bh // 2), max(bw, bh) // 2, color, 2)
            label = "plato c/comida" if it.estado == PLATOS_CON_COMIDA else "plato vacio"
            cv2.putText(out, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
        banner = f"Personas: {'SI' if det.personas else 'NO'}  |  Platos: {det.platos}"
        cv2.rectangle(out, (0, 0), (min(w, 390), 24), (0, 0, 0), -1)
        cv2.putText(out, banner, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out


# ------------------------------------------------------------- 1) heurística OpenCV
class HeuristicDetector(BaseDetector):
    name = "heuristic"

    # ---- parámetros ajustables ----
    HOUGH_PARAM2 = 45          # menor = más círculos (más falsos positivos)
    MIN_R_FRAC = 0.04          # radio mínimo del plato, fracción del ancho del frame
    MAX_R_FRAC = 0.17          # radio máximo del plato
    MIN_PLATE_CONTRAST = 12.0  # distancia Lab mínima borde-plato vs. mesa alrededor
    MAX_RIM_DEV = 16.0         # uniformidad máxima del borde del plato (Lab)
    FOOD_DELTA = 28.0          # distancia Lab al color del borde para considerar "comida"
    FOOD_RATIO = 0.22          # fracción del centro con "comida" para marcar con_comida
    HOG_MIN_WEIGHT = 0.7       # confianza mínima del detector HOG

    def __init__(self, roi: Optional[str] = None, use_hog: Optional[bool] = None):
        super().__init__(roi)
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                f"Tu OpenCV ({cv2.__version__}) no incluye CascadeClassifier/HOGDescriptor "
                "(OpenCV 5.x los eliminó). Instala la serie 4.x:\n"
                "    pip install \"opencv-python-headless>=4.8,<5\"")
        self.use_hog = (os.getenv("HOG_ENABLED", "1") == "1") if use_hog is None else use_hog
        base = cv2.data.haarcascades
        self.face = self._load(base + "haarcascade_frontalface_default.xml")
        self.upper = self._load(base + "haarcascade_upperbody.xml")
        self.hog = None
        if self.use_hog:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    @staticmethod
    def _load(path: str) -> Optional[cv2.CascadeClassifier]:
        c = cv2.CascadeClassifier(path)
        if c.empty():
            print(f"[detector] AVISO: no se pudo cargar {path}; se omite ese clasificador")
            return None
        return c

    # ---- personas ----
    def _detect_people(self, frame: np.ndarray, gray_eq: np.ndarray) -> List[Box]:
        boxes: List[Box] = []
        if self.face is not None:
            for r in self.face.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)):
                boxes.append(tuple(int(v) for v in r))
        if self.upper is not None:
            for r in self.upper.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(70, 70)):
                boxes.append(tuple(int(v) for v in r))
        if self.hog is not None:
            h, w = frame.shape[:2]
            f = 400.0 / w if w > 400 else 1.0           # HOG en una imagen más pequeña: más rápido
            small = cv2.resize(frame, (int(w * f), int(h * f))) if f != 1.0 else frame
            rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
            rects = np.array(rects).reshape(-1, 4)
            weights = np.array(weights).reshape(-1)
            for r, wt in zip(rects, weights):
                if wt >= self.HOG_MIN_WEIGHT:
                    boxes.append(tuple(int(v / f) for v in r))
        boxes = [b for b in boxes if self._in_roi(b, frame.shape)]
        return _merge_boxes(boxes)

    # ---- platos ----
    def _detect_plates(self, frame: np.ndarray, gray: np.ndarray) -> List[PlateItem]:
        h, w = gray.shape
        min_r, max_r = int(w * self.MIN_R_FRAC), int(w * self.MAX_R_FRAC)
        blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 2,
                                   param1=120, param2=self.HOUGH_PARAM2,
                                   minRadius=min_r, maxRadius=max_r)
        if circles is None:
            return []
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        items: List[PlateItem] = []
        for x, y, r in np.round(circles[0][:8]).astype(int):
            if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
                continue                                    # círculo cortado por el borde
            box = (int(x - r), int(y - r), int(2 * r), int(2 * r))
            if not self._in_roi(box, frame.shape):
                continue
            res = self._classify_plate(lab, int(x), int(y), int(r))
            if res is not None:
                estado, score = res
                items.append(PlateItem(box=box, estado=estado, score=score))
        return items

    def _classify_plate(self, lab: np.ndarray, x: int, y: int, r: int) -> Optional[Tuple[str, float]]:
        """None si el círculo no parece un plato; si lo es, (estado, fracción_de_comida)."""
        h, w = lab.shape[:2]
        ro = int(r * 1.5)
        x0, x1, y0, y1 = max(0, x - ro), min(w, x + ro + 1), max(0, y - ro), min(h, y + ro + 1)
        patch = lab[y0:y1, x0:x1].astype(np.float32)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        rim = (d >= 0.78 * r) & (d <= 0.98 * r)          # borde del plato
        center = d <= 0.62 * r                           # zona donde estaría la comida
        outside = (d >= 1.12 * r) & (d <= 1.45 * r)      # mesa alrededor
        if min(rim.sum(), center.sum(), outside.sum()) < 30:
            return None
        rim_col = np.median(patch[rim], axis=0)
        out_col = np.median(patch[outside], axis=0)
        if np.linalg.norm(rim_col - out_col) < self.MIN_PLATE_CONTRAST:
            return None                                   # no contrasta con la mesa
        if np.median(np.linalg.norm(patch[rim] - rim_col, axis=1)) > self.MAX_RIM_DEV:
            return None                                   # borde irregular: no es un plato liso
        dist = np.linalg.norm(patch[center] - rim_col, axis=1)
        ratio = float((dist > self.FOOD_DELTA).mean())
        return (PLATOS_CON_COMIDA if ratio >= self.FOOD_RATIO else PLATOS_VACIOS), ratio

    # ---- API ----
    def detect(self, frame_bgr: np.ndarray) -> Detection:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        people = self._detect_people(frame_bgr, cv2.equalizeHist(gray))
        plates = self._detect_plates(frame_bgr, gray)

        if not plates:
            raw_platos = PLATOS_NINGUNO
        elif any(p.estado == PLATOS_CON_COMIDA for p in plates):
            raw_platos = PLATOS_CON_COMIDA
        else:
            raw_platos = PLATOS_VACIOS

        personas, platos = self._smooth(bool(people), raw_platos)
        return Detection(personas=personas, platos=platos, personas_boxes=people,
                         platos_items=plates, raw_personas=bool(people), raw_platos=raw_platos)


# ----------------------------------------------------------------- 2) YOLO (opcional)
class YoloDetector(BaseDetector):
    """
    Integración con Ultralytics YOLO.

        pip install ultralytics
        set DETECTOR=yolo            (Windows: set / PowerShell: $env:DETECTOR="yolo")
        set YOLO_MODEL=yolov8n.pt    (se descarga solo la primera vez)

    Mapeo por NOMBRE de clase (así funciona con COCO y con modelos propios):
      · persona  : "person" / "persona"
      · comida   : COCO (pizza, sandwich, cake...) o "plato_con_comida" / "food"
      · vajilla  : COCO (bowl, cup, fork, knife, spoon, wine glass) o "plato_vacio" / "plate"
    COCO NO tiene la clase "plato", por eso se usan cubiertos/tazones/vasos como indicio.
    Lo ideal: entrenar un modelo propio (Roboflow/Ultralytics) con las clases
    `persona`, `plato_vacio`, `plato_con_comida` y apuntar YOLO_MODEL=mi_modelo.pt.
    """
    name = "yolo"

    PERSON = {"person", "persona"}
    FOOD = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "plato_con_comida", "con_comida", "food", "comida"}
    DISHES = {"bowl", "cup", "wine glass", "fork", "knife", "spoon", "plate", "plato",
              "plato_vacio", "vacio", "vacios"}

    def __init__(self, model_path: Optional[str] = None, conf: float = 0.35, roi: Optional[str] = None):
        super().__init__(roi)
        try:
            from ultralytics import YOLO  # import perezoso: sólo si eliges este detector
        except ImportError as e:
            raise RuntimeError("Instala ultralytics:  pip install ultralytics") from e
        self.model = YOLO(model_path or os.getenv("YOLO_MODEL", "yolov8n.pt"))
        self.names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        self.conf = float(os.getenv("YOLO_CONF", conf))

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        res = self.model.predict(frame_bgr, conf=self.conf, imgsz=640, verbose=False)[0]
        people: List[Box] = []
        plates: List[PlateItem] = []
        for b in res.boxes:
            name = self.names.get(int(b.cls[0]), "")
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            box = (x1, y1, x2 - x1, y2 - y1)
            if not self._in_roi(box, frame_bgr.shape):
                continue
            score = float(b.conf[0])
            if name in self.PERSON:
                people.append(box)
            elif name in self.FOOD:
                plates.append(PlateItem(box, PLATOS_CON_COMIDA, score))
            elif name in self.DISHES:
                plates.append(PlateItem(box, PLATOS_VACIOS, score))

        if not plates:
            raw_platos = PLATOS_NINGUNO
        elif any(p.estado == PLATOS_CON_COMIDA for p in plates):
            raw_platos = PLATOS_CON_COMIDA
        else:
            raw_platos = PLATOS_VACIOS
        personas, platos = self._smooth(bool(people), raw_platos)
        return Detection(personas, platos, people, plates, bool(people), raw_platos)


# ----------------------------------------------------------------- 3) simulador
class SimDetector(BaseDetector):
    """Recorre los 4 estados cada SIM_PERIOD segundos (ignora la imagen)."""
    name = "sim"
    SEQUENCE = [(False, PLATOS_NINGUNO), (True, PLATOS_NINGUNO),
                (True, PLATOS_CON_COMIDA), (False, PLATOS_CON_COMIDA)]

    def __init__(self, roi: Optional[str] = None, period: Optional[float] = None):
        super().__init__(roi, smooth_window=1)
        self.period = float(os.getenv("SIM_PERIOD", period or 25))
        self._t0 = time.monotonic()

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        h, w = frame_bgr.shape[:2]
        idx = int((time.monotonic() - self._t0) // self.period) % len(self.SEQUENCE)
        personas, platos = self.SEQUENCE[idx]
        pboxes = [(int(w * .12), int(h * .15), int(w * .16), int(h * .35)),
                  (int(w * .70), int(h * .15), int(w * .16), int(h * .35))] if personas else []
        r = int(w * 0.09)
        items = []
        if platos != PLATOS_NINGUNO:
            items = [PlateItem((w // 2 - r, h // 2 - r, 2 * r, 2 * r), platos, 0.5)]
        return Detection(personas, platos, pboxes, items, personas, platos)


# --------------------------------------------------------------------- fábrica
def create_detector(kind: Optional[str] = None) -> BaseDetector:
    kind = (kind or os.getenv("DETECTOR", "heuristic")).lower()
    if kind == "yolo":
        return YoloDetector()
    if kind == "sim":
        return SimDetector()
    if kind == "heuristic":
        return HeuristicDetector()
    raise ValueError(f"DETECTOR desconocido: {kind!r} (usa heuristic | yolo | sim)")
