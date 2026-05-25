"""
Detector de tapaboca (mascarilla facial).
Usa OpenCV haarcascade para detección de rostros y análisis de color/textura
para clasificar si la región inferior de la cara está cubierta.
"""
import cv2
import numpy as np
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Cooldown entre alertas por cámara (segundos) para evitar spam
COOLDOWN_SEGUNDOS = 8
_ultimo_alerta: dict[int, float] = {}


class MaskDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("No se pudo cargar haarcascade_frontalface_default.xml")
        logger.info("[MaskDetector] Cargado correctamente")

    def analizar_frame(self, frame: np.ndarray, camara_id: int) -> dict:
        """
        Detecta rostros y clasifica si cada uno lleva tapaboca.
        Retorna el mismo formato que detector.py para compatibilidad con Laravel.
        """
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
                "sin_tapaboca": 0,
                "tipo_deteccion": "tapaboca",
            }

        rostros = []
        sin_tapaboca_count = 0

        for (x, y, w, h) in faces:
            face_roi = frame[y : y + h, x : x + w]
            tiene_tapaboca, confianza = self._clasificar(face_roi)
            if not tiene_tapaboca:
                sin_tapaboca_count += 1
            rostros.append({
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "tiene_tapaboca": tiene_tapaboca,
                "confianza": round(confianza, 3),
            })

        confianza_max = max(r["confianza"] for r in rostros)
        mensaje = "sin_tapaboca" if sin_tapaboca_count > 0 else "con_tapaboca"

        return {
            "camara_id": camara_id,
            "mensaje": mensaje,
            "identificado": True,
            "persona_id": None,
            "confianza": confianza_max,
            "total_rostros": int(len(faces)),
            "sin_tapaboca": sin_tapaboca_count,
            "tipo_deteccion": "tapaboca",
            "rostros": rostros,
        }

    def _clasificar(self, face_roi: np.ndarray) -> tuple[bool, float]:
        """
        Heurístico de 3 señales para detectar tapaboca:
        1. Ratio de piel en región inferior (nariz-barbilla) — baja piel → tapaboca
        2. Uniformidad de color (tapabocas son colores uniformes)
        3. Varianza de textura (piel tiene más textura que tela)
        """
        h, w = face_roi.shape[:2]
        if h < 30 or w < 30:
            return False, 0.5

        # Zona inferior de la cara (45%–100% vertical)
        lower = face_roi[int(h * 0.45) :, :]

        skin_ratio   = self._ratio_piel(lower)
        uniformidad  = self._uniformidad_color(lower)
        textura      = self._varianza_textura(lower)

        # Score 0→1 donde 1 = definitivamente tiene tapaboca
        score = (
            (1.0 - min(skin_ratio / 0.40, 1.0)) * 0.50
            + (uniformidad / 100.0)              * 0.30
            + (1.0 - min(textura / 25.0, 1.0))  * 0.20
        )

        tiene = score > 0.55
        confianza = score if tiene else (1.0 - score)
        return tiene, float(np.clip(confianza, 0.0, 1.0))

    def _ratio_piel(self, region: np.ndarray) -> float:
        """Porcentaje de píxeles con tono de piel (YCrCb — válido para múltiples tonos)."""
        if region.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        return float(np.count_nonzero(mask)) / mask.size

    def _uniformidad_color(self, region: np.ndarray) -> float:
        """Uniformidad: baja desviación en matiz y saturación = color uniforme = tapaboca."""
        if region.size == 0:
            return 0.0
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        std_h = float(np.std(hsv[:, :, 0]))
        std_s = float(np.std(hsv[:, :, 1]))
        return float(max(0.0, 100.0 - (std_h + std_s * 0.5)))

    def _varianza_textura(self, region: np.ndarray) -> float:
        """Varianza del Laplaciano — alta = mucha textura (piel), baja = superficie lisa (tapaboca)."""
        if region.size == 0:
            return 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(lap))


# ── Instancia global ──────────────────────────────────────
mask_detector = MaskDetector()


def analizar_frame_tapaboca(frame: np.ndarray, camara_id: int) -> dict:
    """Wrapper público — mismo contrato que detector.analizar_frame."""
    return mask_detector.analizar_frame(frame, camara_id)


def debe_enviar_alerta(camara_id: int) -> bool:
    """Cooldown: evita enviar alertas de tapaboca más de 1 vez cada COOLDOWN_SEGUNDOS."""
    ahora = time.time()
    ultimo = _ultimo_alerta.get(camara_id, 0)
    if ahora - ultimo >= COOLDOWN_SEGUNDOS:
        _ultimo_alerta[camara_id] = ahora
        return True
    return False
