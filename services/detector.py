"""
Servicio de detección facial usando DeepFace.
Compara rostros contra la base de datos local (faces_db/).
"""
import os
import cv2
import numpy as np
import httpx
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from deepface import DeepFace

logger = logging.getLogger(__name__)

FACES_DB   = os.getenv("FACES_DB_PATH", "./faces_db")
MODEL      = os.getenv("DEEPFACE_MODEL", "Facenet512")
DETECTOR   = os.getenv("DEEPFACE_DETECTOR", "opencv")
THRESHOLD  = float(os.getenv("DEEPFACE_THRESHOLD", "0.40"))
LARAVEL    = os.getenv("LARAVEL_URL", "http://localhost:8000")
TOKEN      = os.getenv("LARAVEL_SERVICE_TOKEN", "")

_STORAGE_ROOT = Path(os.getenv(
    "LARAVEL_STORAGE_PATH",
    "C:/Users/Nestor/Desktop/vigilancia-facial/storage/app/public",
))
CAPTURAS_DIR = _STORAGE_ROOT / "capturas"
CAPTURAS_DIR.mkdir(parents=True, exist_ok=True)


def analizar_frame(frame: np.ndarray, camara_id: int) -> dict:
    """
    Analiza un frame de OpenCV.
    Retorna dict con resultado listo para enviarse a Laravel.
    """
    captura_path = _guardar_captura(frame, camara_id)

    # ¿Hay rostros en el frame?
    try:
        rostros = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR,
            enforce_detection=True,
        )
    except Exception:
        # No se detectó ningún rostro
        return {
            "camara_id": camara_id,
            "identificado": False,
            "persona_id": None,
            "confianza": 0.0,
            "captura": captura_path,
            "embedding": None,
            "mensaje": "sin_rostro",
        }

    if not rostros:
        return {
            "camara_id": camara_id,
            "identificado": False,
            "persona_id": None,
            "confianza": 0.0,
            "captura": captura_path,
            "embedding": None,
            "mensaje": "sin_rostro",
        }

    # Buscar en la base de datos de rostros registrados
    db_path = Path(FACES_DB)
    if not db_path.exists() or not any(db_path.iterdir()):
        return {
            "camara_id": camara_id,
            "identificado": False,
            "persona_id": None,
            "confianza": 0.5,
            "captura": captura_path,
            "embedding": None,
            "mensaje": "db_vacia",
        }

    try:
        resultados = DeepFace.find(
            img_path=frame,
            db_path=str(db_path),
            model_name=MODEL,
            detector_backend=DETECTOR,
            distance_metric="cosine",
            enforce_detection=False,
            silent=True,
        )
    except Exception as e:
        logger.error(f"Error en DeepFace.find: {e}")
        return {
            "camara_id": camara_id,
            "identificado": False,
            "persona_id": None,
            "confianza": 0.0,
            "captura": captura_path,
            "embedding": None,
            "mensaje": f"error_find: {str(e)}",
        }

    # DeepFace.find retorna lista de DataFrames (uno por rostro en la imagen)
    if not resultados or len(resultados[0]) == 0:
        return {
            "camara_id": camara_id,
            "identificado": False,
            "persona_id": None,
            "confianza": 0.5,
            "captura": captura_path,
            "embedding": None,
            "mensaje": "no_coincide",
        }

    mejor = resultados[0].iloc[0]
    distancia = float(mejor.get(f"{MODEL}_cosine", 1.0))
    confianza = max(0.0, 1.0 - distancia)

    if distancia > THRESHOLD:
        return {
            "camara_id": camara_id,
            "identificado": False,
            "persona_id": None,
            "confianza": round(confianza, 3),
            "captura": captura_path,
            "embedding": None,
            "mensaje": "baja_confianza",
        }

    # Extraer persona_id del path: faces_db/persona_42/foto.jpg → 42
    identity_path = mejor.get("identity", "")
    persona_id = _extraer_persona_id(identity_path)

    # Calcular embedding del rostro detectado
    embedding = _get_embedding(frame)

    return {
        "camara_id": camara_id,
        "identificado": True,
        "persona_id": persona_id,
        "confianza": round(confianza, 3),
        "captura": captura_path,
        "embedding": embedding,
        "mensaje": "identificado",
    }


async def enviar_resultado_a_laravel(resultado: dict) -> bool:
    """Llama a POST /api/facial/procesar en Laravel con el resultado."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{LARAVEL}/api/facial/procesar",
                json=resultado,
                headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"},
            )
            return resp.status_code == 200
    except Exception as e:
        logger.error(f"Error enviando resultado a Laravel: {e}")
        return False


def registrar_rostro(persona_id: int, imagen: np.ndarray) -> str:
    """
    Guarda la foto de referencia en faces_db/persona_{id}/.
    DeepFace usará esta carpeta para reconocimiento.
    """
    carpeta = Path(FACES_DB) / f"persona_{persona_id}"
    carpeta.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = carpeta / f"ref_{ts}.jpg"
    cv2.imwrite(str(ruta), imagen)

    # Invalidar representaciones cacheadas de DeepFace
    pkl = Path(FACES_DB) / "representations_facenet512.pkl"
    if pkl.exists():
        pkl.unlink()

    return str(ruta)


def guardar_captura(frame: np.ndarray, camara_id: int) -> Optional[str]:
    """Guarda el frame en el storage público de Laravel y retorna la ruta relativa (capturas/filename.jpg)."""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"cam{camara_id}_{ts}.jpg"
        cv2.imwrite(str(CAPTURAS_DIR / filename), frame)
        return f"capturas/{filename}"
    except Exception:
        return None


def _guardar_captura(frame: np.ndarray, camara_id: int) -> Optional[str]:
    return guardar_captura(frame, camara_id)


def _extraer_persona_id(identity_path: str) -> Optional[int]:
    """faces_db/persona_42/foto.jpg → 42"""
    try:
        partes = Path(identity_path).parts
        for parte in partes:
            if parte.startswith("persona_"):
                return int(parte.replace("persona_", ""))
    except Exception:
        pass
    return None


def _get_embedding(frame: np.ndarray) -> Optional[list]:
    try:
        resultado = DeepFace.represent(
            img_path=frame,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False,
        )
        if resultado:
            return resultado[0].get("embedding")
    except Exception:
        pass
    return None
