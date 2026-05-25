"""
Detector de casco de seguridad (hardhat).
Analiza la región SOBRE el rostro detectado buscando la silueta y color típico de un casco.
Heurístico basado en:
  - Extensión vertical sobre la cabeza (el casco sobresale hacia arriba)
  - Colores característicos: amarillo, blanco, naranja, azul, rojo (HSV)
  - Forma convexa / área compacta sobre la frente
"""
import cv2
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

COOLDOWN_SEGUNDOS = 8
_ultimo_alerta: dict[int, float] = {}

# Rangos HSV de colores típicos de cascos industriales
HELMET_COLOR_RANGES = [
    # Amarillo
    ((20,  80,  80), (35, 255, 255)),
    # Naranja
    ((5,  100,  80), (20, 255, 255)),
    # Blanco
    ((0,    0, 180), (180,  40, 255)),
    # Azul
    ((95,  60,  60), (130, 255, 255)),
    # Rojo (parte 1)
    ((0,  100,  80), (8,  255, 255)),
    # Rojo (parte 2)
    ((170, 100,  80), (180, 255, 255)),
]


class HelmetDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("[HelmetDetector] Cargado correctamente")

    def analizar_frame(self, frame: np.ndarray, camara_id: int) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return {
                "camara_id": camara_id,
                "mensaje": "sin_rostro",
                "identificado": False,
                "persona_id": None,
                "confianza": 0.0,
                "total_rostros": 0,
                "sin_casco": 0,
                "tipo_deteccion": "casco",
            }

        rostros = []
        sin_casco_count = 0
        frame_h, frame_w = frame.shape[:2]

        for (x, y, w, h) in faces:
            # Región sobre la cabeza: desde (y - h*0.8) hasta y, misma anchura expandida
            head_top = max(0, y - int(h * 0.9))
            head_x1  = max(0, x - int(w * 0.15))
            head_x2  = min(frame_w, x + w + int(w * 0.15))
            head_region = frame[head_top:y, head_x1:head_x2]

            tiene_casco, confianza = self._detectar_casco(head_region, w, h)

            if not tiene_casco:
                sin_casco_count += 1

            rostros.append({
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "tiene_casco": tiene_casco,
                "confianza": round(confianza, 3),
            })

        confianza_max = max(r["confianza"] for r in rostros)
        mensaje = "sin_casco" if sin_casco_count > 0 else "con_casco"

        return {
            "camara_id": camara_id,
            "mensaje": mensaje,
            "identificado": True,
            "persona_id": None,
            "confianza": confianza_max,
            "total_rostros": int(len(faces)),
            "sin_casco": sin_casco_count,
            "tipo_deteccion": "casco",
            "rostros": rostros,
        }

    def _detectar_casco(self, head_region: np.ndarray, face_w: int, face_h: int) -> tuple[bool, float]:
        if head_region.size == 0 or head_region.shape[0] < 10:
            # Sin espacio sobre la cabeza → no hay casco visible
            return False, 0.75

        # Señal 1: ¿hay píxeles de color casco sobre la frente?
        ratio_color = self._ratio_color_casco(head_region)

        # Señal 2: altura de la región de color (casco tiene presencia vertical)
        altura_relativa = head_region.shape[0] / max(face_h, 1)

        # Señal 3: densidad de color en banda superior
        banda_superior = head_region[: head_region.shape[0] // 2, :]
        ratio_superior = self._ratio_color_casco(banda_superior) if banda_superior.size > 0 else 0.0

        # Score: alto color en región superior + altura suficiente → casco
        score_casco = (
            min(ratio_color / 0.20, 1.0)   * 0.50
            + min(ratio_superior / 0.15, 1.0) * 0.30
            + min(altura_relativa / 0.60, 1.0) * 0.20
        )

        tiene_casco = score_casco > 0.45
        confianza = score_casco if tiene_casco else (1.0 - score_casco)
        return tiene_casco, float(np.clip(confianza, 0.0, 1.0))

    def _ratio_color_casco(self, region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in HELMET_COLOR_RANGES:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        return float(np.count_nonzero(mask)) / mask.size


helmet_detector = HelmetDetector()


def analizar_frame_casco(frame: np.ndarray, camara_id: int) -> dict:
    return helmet_detector.analizar_frame(frame, camara_id)


def debe_enviar_alerta_casco(camara_id: int) -> bool:
    ahora = time.time()
    if ahora - _ultimo_alerta.get(camara_id, 0) >= COOLDOWN_SEGUNDOS:
        _ultimo_alerta[camara_id] = ahora
        return True
    return False
