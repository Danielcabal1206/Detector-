# 🍽️ Restaurant Monitor — monitoreo de mesas con iPhone + PC

Prototipo de arquitectura híbrida: el **iPhone** es la cámara, la **PC** es el servidor
(visión por computador + máquina de estados) y el centro de alertas.

```
 iPhone (Safari)                        PC (Python)                          PC (navegador)
┌──────────────────┐   JPEG / WebSocket   ┌─────────────────────────┐  JSON + frame  ┌────────────────┐
│  camera.html     │ ───────────────────▶ │ main.py (FastAPI)       │ ─────────────▶ │ dashboard.html │
│  getUserMedia    │      /ws/camera      │  ├ detector.py  (OpenCV)│  /ws/dashboard │ video + estado │
│  cámara trasera  │                      │  └ automaton.py (FSM)   │                │ + alertas      │
└──────────────────┘                      └─────────────────────────┘                └────────────────┘
```

## Los 4 estados

| # | Estado | Personas | Platos | Alerta |
|---|--------|:--------:|--------|--------|
| 1 | Mesa vacía | no | ninguno | — (estado natural) |
| 2 | Mesa con restos | no | vacíos / con comida | 🔴 **Limpieza** si dura ≥ `DIRTY_ALERT_SECONDS` |
| 3 | Clientes sin comida | sí | ninguno | 🟠 **Atención** si dura ≥ `WAITING_ALERT_SECONDS` |
| 4 | Clientes comiendo | sí | vacíos / con comida | — (disfrutando) |

Reglas del autómata (`backend/automaton.py`):
- **Debounce**: un estado nuevo debe observarse `DEBOUNCE_SECONDS` seguidos antes de aceptarse (evita parpadeos del detector). El cronómetro cuenta desde que la condición *empezó*.
- Cada alerta se dispara **una vez** y se resuelve sola cuando cambia el estado.
- Si la cámara se cae más de 10 s, al volver se reinician los cronómetros (no hay alertas falsas con datos viejos).
- Por defecto los umbrales son cortos (15 s / 20 s) para que la demo sea visible. **En un restaurante real** usa algo como 120–300 s y 180–600 s (ver variables más abajo o el panel del dashboard).

## Estructura

```
restaurant-monitor/
├── backend/
│   ├── main.py             # FastAPI + WebSockets + API REST
│   ├── detector.py         # Visión: heurística OpenCV | YOLO | simulador
│   ├── automaton.py        # Máquina de estados (4 estados + alertas temporales)
│   ├── make_cert.py        # Genera el certificado HTTPS autofirmado (necesario para iPhone)
│   ├── test_automaton.py   # Pruebas del autómata con reloj simulado
│   └── requirements.txt
├── frontend/
│   ├── camera.html         # Vista del iPhone (captura y envía video)
│   └── dashboard.html      # Vista de la PC (video procesado, estado, alertas)
└── README.md
```

## 1. Instalación (PC)

Requiere **Python 3.10+**.

**Windows (PowerShell)**
```powershell
cd restaurant-monitor\backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
cd restaurant-monitor/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` fija OpenCV en la serie **4.x** a propósito: OpenCV 5.0 eliminó
> `CascadeClassifier` y `HOGDescriptor`, que el detector heurístico necesita.

## 2. Probar primero SIN cámara (2 minutos)

El modo `sim` recorre los 4 estados solo, para ver el dashboard y las alertas funcionando.

| Terminal | Comando |
|----------|---------|
| PowerShell | `$env:DETECTOR="sim"; python main.py` |
| cmd.exe | `set DETECTOR=sim && python main.py` |
| macOS / Linux | `DETECTOR=sim python main.py` |

Abre `http://localhost:8000/dashboard`. Como el simulador ignora la imagen, necesita recibir
fotogramas: abre en otra pestaña `http://localhost:8000/camera` (con la webcam de la PC) y pulsa
*Iniciar transmisión*. El simulador cambia de estado cada 25 s (`SIM_PERIOD`).

## 3. Conectar el iPhone (modo real)

**Requisitos de red:** PC y iPhone en la **misma red Wi-Fi** (no uses la red de invitados, suele aislar dispositivos).

### 3.1 Certificado HTTPS (obligatorio)
Safari **solo permite usar la cámara en páginas HTTPS** (o `localhost`). Como el iPhone entra por
la IP de la PC, hace falta HTTPS:

```bash
cd backend
python make_cert.py          # crea backend/certs/cert.pem y key.pem con las IPs de tu PC
python main.py               # al detectar certs/, arranca en HTTPS y te imprime las URLs
```

Si cambias de red o de IP, vuelve a ejecutar `python make_cert.py`.

### 3.2 Encontrar la IP de la PC
`main.py` ya imprime la URL al arrancar (`Cámara (iPhone): https://192.168.x.x:8000/camera`).
Si quieres comprobarla a mano:
- Windows: `ipconfig` → "Dirección IPv4" del adaptador Wi-Fi
- macOS: `ipconfig getifaddr en0`
- Linux: `hostname -I`

### 3.3 Abrir la cámara en el iPhone
1. En Safari abre `https://<IP-DE-LA-PC>:8000/camera`.
2. Aparecerá *"Esta conexión no es privada"* (es tu certificado autofirmado): **Mostrar detalles → visitar este sitio web**.
3. Pulsa **Iniciar transmisión** y acepta el permiso de cámara. Se usa la **cámara trasera**.
4. Apoya el iPhone apuntando a la mesa, **con la pantalla encendida** y cargando.
5. En la PC abre `https://localhost:8000/dashboard` (acepta también la advertencia del certificado).

### Si algo no conecta
| Síntoma | Causa / solución |
|---------|------------------|
| El iPhone no carga la página | Firewall de la PC. **Windows:** permite Python en "Redes privadas" (o `netsh advfirewall firewall add rule name="monitor" dir=in action=allow protocol=TCP localport=8000`). **macOS:** acepta el aviso de conexiones entrantes. Verifica misma Wi-Fi. |
| "El navegador no permite usar la cámara" | Estás en `http://`. Genera el certificado y entra por `https://`. |
| Permiso de cámara denegado | iPhone: Ajustes › Safari › Cámara → *Preguntar/Permitir*. |
| El WebSocket seguro no conecta | Safari a veces rechaza `wss://` con certificados autofirmados. `camera.html` lo detecta y **cambia sola a HTTP POST** (`/api/frame`), que funciona igual (verás "Transporte: HTTP POST"). Para WebSocket real, instala un certificado de confianza (p. ej. con [mkcert](https://github.com/FiloSottile/mkcert) → pon `cert.pem`/`key.pem` en `backend/certs/`). |
| El video se corta | Safari pausa el envío si se bloquea la pantalla o cambias de pestaña. Mantén la pestaña visible. |

## 4. Detector: heurística, YOLO o simulador

Cambia con la variable `DETECTOR` (`heuristic` por defecto | `yolo` | `sim`).

### `heuristic` (OpenCV clásico, sin descargas)
- **Personas:** cascadas Haar (rostro + torso) y HOG de peatones (`HOG_ENABLED=0` lo apaga y acelera).
- **Platos:** círculos de Hough que contrastan con la mesa y tienen borde uniforme; si el centro difiere del color del borde → `con_comida`, si no → `vacios`.
- Suavizado de 5 fotogramas + debounce del autómata. ~150 ms por fotograma en CPU.
- Ajustes: constantes al inicio de `HeuristicDetector` (`FOOD_RATIO`, `FOOD_DELTA`, `HOUGH_PARAM2`…).

**Limitaciones honestas:** es un prototipo. Funciona mejor con buena luz, platos redondos y lisos de color claro sobre un mantel de otro color, y gente mirando hacia la cámara. Falla con platos estampados o muy parecidos al mantel, reflejos fuertes, o personas de espaldas/de perfil. Para uso real, pasa a YOLO.

### `yolo` (recomendado para producción)
```bash
pip install ultralytics
# PowerShell:  $env:DETECTOR="yolo"; python main.py
DETECTOR=yolo YOLO_MODEL=yolov8n.pt python main.py     # COCO preentrenado, se descarga solo
```
- Con COCO, `person` → personas; `pizza/sandwich/cake…` → `con_comida`; `bowl/cup/fork/knife/spoon/wine glass` → `vacios`. COCO **no tiene clase "plato"**, así que es una aproximación.
- **Lo ideal:** entrenar un modelo propio (Roboflow / Ultralytics) con las clases `persona`, `plato_vacio`, `plato_con_comida` y usar `YOLO_MODEL=mi_modelo.pt`. El mapeo es **por nombre de clase**, no hay que tocar código.
- Para integrar otro modelo, crea una subclase de `BaseDetector` con un método `detect(frame) -> Detection` y regístrala en `create_detector()`.

### Región de interés (muy recomendable)
Para ignorar al personal que pasa por el pasillo, define la zona de la mesa (valores 0–1: `x,y,ancho,alto`):
```bash
TABLE_ROI="0.2,0.3,0.6,0.5" python main.py
```
Se dibuja como un rectángulo amarillo en el video del dashboard.

## 5. Variables de entorno

| Variable | Defecto | Descripción |
|----------|---------|-------------|
| `DETECTOR` | `heuristic` | `heuristic` \| `yolo` \| `sim` |
| `DIRTY_ALERT_SECONDS` | `15` | Estado 2 → alerta de limpieza |
| `WAITING_ALERT_SECONDS` | `20` | Estado 3 → alerta de atención |
| `DEBOUNCE_SECONDS` | `2` | Estabilidad para cambiar de estado |
| `TABLE_ROI` | — | Región de interés normalizada |
| `HOG_ENABLED` | `1` | `0` desactiva HOG (más rápido) |
| `YOLO_MODEL` / `YOLO_CONF` | `yolov8n.pt` / `0.35` | Modelo y confianza de YOLO |
| `SIM_PERIOD` | `25` | Segundos por estado en modo `sim` |
| `PROC_WIDTH` | `640` | Ancho de análisis |
| `PORT` | `8000` | Puerto |

Los tres umbrales también se pueden cambiar **en caliente** desde el panel del dashboard.

## 6. API

| Endpoint | Descripción |
|----------|-------------|
| `WS /ws/camera` | Entrada: JPEG binarios desde el iPhone |
| `WS /ws/dashboard` | Salida: `init`, `frame` (imagen base64 + estado + eventos), `config`, `reset` |
| `POST /api/frame` | Alternativa HTTP al WebSocket (cuerpo = JPEG) |
| `GET /api/state` · `/api/events` · `/api/health` | Estado actual, historial, diagnóstico |
| `GET/POST /api/config` | Leer / cambiar umbrales |
| `POST /api/reset` | Reinicia mesa y registro |
| `GET /docs` | Documentación interactiva (Swagger) |

## 7. Pruebas

```bash
cd backend
python test_automaton.py     # 10 pruebas de la máquina de estados con reloj simulado
```

## 8. Siguientes pasos naturales
- **Varias mesas / varias cámaras:** un autómata por `table_id` (hoy hay uno solo).
- **Notificaciones reales:** push/WhatsApp/Telegram al disparar una alerta (el punto de enganche es `machine.update()` en `handle_frame`).
- **Privacidad:** el sistema no guarda video; solo retransmite fotogramas en vivo dentro de tu red local. Si lo usas en un negocio, informa a clientes y personal.
- Autenticación en el dashboard si se expone fuera de la red local.
