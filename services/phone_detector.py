"""
Detector de celular en mano.
Tecnologias:
  - YOLOv8 (ultralytics)  -- deteccion del dispositivo celular (clase 67 COCO)
  - OpenCV YCrCb           -- verificacion de piel/mano sosteniendo el objeto
  - DeepFace (detector.py) -- identificacion facial del portador
  - Timer en memoria       -- registro de inicio / fin de sesion de uso

Nota: MediaPipe 0.10+ elimino la API solutions; se usa deteccion de piel
por color YCrCb que es mas ligera y no requiere dependencias adicionales.
"""
import cv2
import numpy as np
import logging
import time
import httpx
import os
from datetime import datetime
from typing import Optional

from services.detector import guardar_captura, analizar_frame as analizar_facial

logger = logging.getLogger(__name__)

FRAMES_SIN_CELULAR_CERRAR = int(os.getenv("FRAMES_SIN_CELULAR_CERRAR", "5"))
CELULAR_CONF_MIN = float(os.getenv("CELULAR_CONF_MIN", "0.40"))
# Porcentaje minimo de pixeles de piel alrededor del celular para considerar "en mano"
SKIN_RATIO_MIN = float(os.getenv("CELULAR_SKIN_RATIO", "0.08"))
LARAVEL = os.getenv("LARAVEL_URL", "http://localhost:8000")
TOKEN   = os.getenv("LARAVEL_SERVICE_TOKEN", "")

CELL_PHONE_CLASS = 67  # indice en COCO dataset

# Sesiones activas: camara_id -> dict con datos de la sesion
_sesiones: dict[int, dict] = {}

# Modelo YOLO con lazy init
_yolo_model = None


def _yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
        logger.info("[PhoneDetector] YOLOv8n cargado")
    return _yolo_model


# -- Deteccion con YOLOv8 -----------------------------------------------

def _detectar_telefonos(frame: np.ndarray) -> list[dict]:
    """Retorna lista de telefonos detectados con bbox y confianza."""
    results = _yolo()(frame, verbose=False, classes=[CELL_PHONE_CLASS], conf=CELULAR_CONF_MIN)
    phones = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            phones.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confianza": float(box.conf[0]),
            })
    return phones


# -- Verificacion de mano por deteccion de piel (YCrCb) -----------------

def _mano_sostiene(frame: np.ndarray, phones: list[dict]) -> bool:
    """
    Verifica que hay piel humana (mano) en la region alrededor del celular.
    Usa deteccion de color YCrCb (valida para multiples tonos de piel).
    Retorna True si la proporcion de pixeles de piel supera SKIN_RATIO_MIN.
    """
    h_frame, w_frame = frame.shape[:2]
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(
        ycrcb,
        np.array([0,   133, 77],  dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    for phone in phones:
        pad = max(phone["x2"] - phone["x1"], phone["y2"] - phone["y1"]) // 2
        x1 = max(0, phone["x1"] - pad)
        y1 = max(0, phone["y1"] - pad)
        x2 = min(w_frame, phone["x2"] + pad)
        y2 = min(h_frame, phone["y2"] + pad)

        region = skin_mask[y1:y2, x1:x2]
        if region.size == 0:
            continue
        ratio = float(np.count_nonzero(region)) / region.size
        logger.debug(f"[PhoneDetector] skin_ratio={ratio:.3f} (min={SKIN_RATIO_MIN})")
        if ratio >= SKIN_RATIO_MIN:
            return True
    return False


# -- Analisis principal ---------------------------------------------------

def analizar_frame_celular(frame: np.ndarray, camara_id: int) -> dict:
    """
    Analiza un frame buscando celular sostenido con la mano.
    Gestiona sesiones de uso y retorna eventos inicio/fin/ongoing.
    """
    phones = _detectar_telefonos(frame)

    if not phones or not _mano_sostiene(frame, phones):
        return _sin_celular(camara_id)

    confianza_celular = max(p["confianza"] for p in phones)

    resultado_facial = analizar_facial(frame, camara_id)
    persona_id       = resultado_facial.get("persona_id")
    confianza_facial = resultado_facial.get("confianza", 0.0)
    foto_rostro      = resultado_facial.get("captura") or guardar_captura(frame, camara_id)

    ahora = time.time()

    if camara_id not in _sesiones:
        _sesiones[camara_id] = {
            "start_time":      ahora,
            "persona_id":      persona_id,
            "foto_rostro":     foto_rostro,
            "confianza_facial":  confianza_facial,
            "confianza_celular": confianza_celular,
            "frames_sin_celular": 0,
            "session_id":      None,
        }
        logger.info(f"[PhoneDetector] Sesion iniciada cam={camara_id} persona={persona_id} conf={confianza_celular:.2f}")
        return {
            "camara_id":        camara_id,
            "mensaje":          "celular_detectado_inicio",
            "tipo_deteccion":   "celular",
            "identificado":     persona_id is not None,
            "persona_id":       persona_id,
            "confianza":        round(confianza_celular, 3),
            "confianza_facial": round(confianza_facial, 3),
            "foto_rostro":      foto_rostro,
            "inicio":           datetime.fromtimestamp(ahora).isoformat(),
            "es_nuevo":         True,
        }

    sesion = _sesiones[camara_id]
    sesion["frames_sin_celular"] = 0
    if persona_id and not sesion["persona_id"]:
        sesion["persona_id"]       = persona_id
        sesion["confianza_facial"] = confianza_facial
        sesion["foto_rostro"]      = foto_rostro

    return {
        "camara_id":         camara_id,
        "mensaje":           "celular_en_mano",
        "tipo_deteccion":    "celular",
        "identificado":      sesion["persona_id"] is not None,
        "persona_id":        sesion["persona_id"],
        "confianza":         round(confianza_celular, 3),
        "duracion_segundos": int(ahora - sesion["start_time"]),
        "es_nuevo":          False,
    }


def _sin_celular(camara_id: int) -> dict:
    """Sin celular detectado; puede cerrar sesion activa si supera el umbral."""
    if camara_id not in _sesiones:
        return {
            "camara_id":      camara_id,
            "mensaje":        "sin_celular",
            "tipo_deteccion": "celular",
            "identificado":   False,
            "persona_id":     None,
            "confianza":      0.0,
        }

    sesion = _sesiones[camara_id]
    sesion["frames_sin_celular"] = sesion.get("frames_sin_celular", 0) + 1

    if sesion["frames_sin_celular"] < FRAMES_SIN_CELULAR_CERRAR:
        return {
            "camara_id":      camara_id,
            "mensaje":        "sin_celular",
            "tipo_deteccion": "celular",
            "identificado":   False,
            "persona_id":     None,
            "confianza":      0.0,
        }

    fin_time = time.time()
    duracion  = int(fin_time - sesion["start_time"])
    resultado = {
        "camara_id":         camara_id,
        "mensaje":           "celular_sesion_finalizada",
        "tipo_deteccion":    "celular",
        "identificado":      sesion["persona_id"] is not None,
        "persona_id":        sesion["persona_id"],
        "confianza":         sesion["confianza_celular"],
        "confianza_facial":  sesion["confianza_facial"],
        "foto_rostro":       sesion["foto_rostro"],
        "inicio":            datetime.fromtimestamp(sesion["start_time"]).isoformat(),
        "fin":               datetime.fromtimestamp(fin_time).isoformat(),
        "duracion_segundos": duracion,
        "session_id":        sesion.get("session_id"),
        "finalizado":        True,
    }
    del _sesiones[camara_id]
    logger.info(f"[PhoneDetector] Sesion finalizada cam={camara_id} duracion={duracion}s")
    return resultado


def set_session_id(camara_id: int, session_id: int):
    """Guarda el ID del registro Laravel para referenciarlo al cerrar la sesion."""
    if camara_id in _sesiones:
        _sesiones[camara_id]["session_id"] = session_id


# -- Envio a Laravel -------------------------------------------------------

async def enviar_celular_a_laravel(resultado: dict) -> Optional[int]:
    """POST /api/celular/procesar. Retorna session_id si el servidor lo devuelve."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{LARAVEL}/api/celular/procesar",
                json=resultado,
                headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"},
            )
            if resp.status_code in (200, 201):
                return resp.json().get("session_id")
    except Exception as e:
        logger.error(f"[PhoneDetector] Error enviando a Laravel: {e}")
    return None
