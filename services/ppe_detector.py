"""
Detector de EPP por persona (Fase 2).

1. Detecta personas con YOLOv8 COCO (clase 0), que es fiable.
2. Detecta elementos de EPP con el modelo configurado en modelos_epp/config.json.
3. Para cada persona y cada elemento exigido decide:
     presente     → el elemento aparece en la zona del cuerpo que le corresponde
     falta        → la zona se ve y el elemento no aparece (o aparece su clase "no_*")
     no_visible   → la zona no se ve (persona cortada por el encuadre o demasiado pequeña)
   No se depende de las clases "no_*": los modelos públicos casi no las detectan.
4. En streams, una falta solo se reporta si aparece en 'minimo' de los últimos 'ventana' frames.

El modelo es intercambiable: basta con cambiar config.json.
"""
import json
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import httpx
import numpy as np

from services.detector import guardar_captura

logger = logging.getLogger(__name__)

DIR_MODELOS = Path(__file__).resolve().parent.parent / "modelos_epp"

_lock = Lock()
_cfg: Optional[dict] = None
_modelo_epp = None
_modelo_personas = None
_error_carga: Optional[str] = None


def _cargar() -> bool:
    """Carga perezosa del config y de ambos modelos. Devuelve False si no hay modelo de EPP utilizable."""
    global _cfg, _modelo_epp, _modelo_personas, _error_carga
    with _lock:
        if _modelo_epp is not None:
            return True
        try:
            from ultralytics import YOLO
            _cfg = json.loads((DIR_MODELOS / "config.json").read_text(encoding="utf-8"))
            ruta = DIR_MODELOS / _cfg["archivo"]
            if not ruta.exists():
                raise FileNotFoundError(f"No existe el modelo {ruta.name}")
            _modelo_epp = YOLO(str(ruta))
            _modelo_personas = YOLO(str(Path(__file__).resolve().parent.parent / "yolov8n.pt"))
            _error_carga = None
            logger.info(f"[EPP] Modelo cargado: {_cfg['nombre']}")
            return True
        except Exception as e:
            _error_carga = str(e)
            _modelo_epp = None
            logger.error(f"[EPP] No se pudo cargar el modelo de EPP: {e}")
            return False


def recargar() -> bool:
    """Descarta el modelo en memoria y vuelve a cargar el de config.json (tras activar un candidato)."""
    global _cfg, _modelo_epp
    with _lock:
        _cfg = None
        _modelo_epp = None
    return _cargar()


def capacidades() -> dict:
    """Qué elementos sabe verificar el modelo de EPP (para el catálogo de Laravel)."""
    ok = _cargar()
    if not ok:
        return {"disponible": False, "error": _error_carga, "elementos": []}
    elementos = sorted({c["elemento"] for c in _cfg["clases"].values()})
    return {
        "disponible": True,
        "modelo": _cfg["nombre"],
        "id": _cfg.get("id"),
        "apto_produccion": bool(_cfg.get("apto_produccion")),
        "advertencia": _cfg.get("advertencia"),
        "elementos": elementos,
        "clases": list(_modelo_epp.names.values()),
    }


# ── Análisis de un frame ──────────────────────────────────────

def _solapamiento(caja: list, persona: list, franja: tuple) -> float:
    """Fracción del recuadro del elemento que cae dentro de la franja vertical de la persona."""
    px1, py1, px2, py2 = persona
    alto = py2 - py1
    fy1, fy2 = py1 + franja[0] * alto, py1 + franja[1] * alto
    # Margen horizontal: los elementos (guantes, botas) pueden salir un poco del recuadro
    margen = (px2 - px1) * 0.15
    x1, y1, x2, y2 = caja
    ix = max(0, min(x2, px2 + margen) - max(x1, px1 - margen))
    iy = max(0, min(y2, fy2) - max(y1, fy1))
    area = max(1, (x2 - x1) * (y2 - y1))
    return ix * iy / area


def _zona_visible(persona: list, franja: tuple, alto_img: int, ancho_img: int, alto_min: int) -> bool:
    x1, y1, x2, y2 = persona
    if (y2 - y1) < alto_min:
        return False                 # persona demasiado pequeña para verificar
    borde = 4
    if franja[1] >= 0.95 and y2 >= alto_img - borde:
        return False                 # pies cortados por el borde inferior
    if franja[0] <= 0.05 and y1 <= borde:
        return False                 # cabeza cortada por el borde superior
    return True


def analizar(frame: np.ndarray, requeridos: list[str]) -> dict:
    """Evalúa los elementos requeridos para cada persona del frame."""
    if not _cargar():
        return {"disponible": False, "error": _error_carga, "personas": []}

    alto, ancho = frame.shape[:2]
    conf_min = float(_cfg.get("confianza_minima", 0.25))
    zonas = _cfg["zonas_cuerpo"]
    clases = _cfg["clases"]

    personas = [
        [float(v) for v in b] for b in
        _modelo_personas.predict(frame, classes=[0], conf=0.4, verbose=False)[0].boxes.xyxy.tolist()
    ]
    r = _modelo_epp.predict(frame, conf=conf_min, verbose=False)[0]
    detecciones = []
    for caja, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        nombre = _modelo_epp.names[int(cls)]
        mapa = clases.get(nombre)
        if mapa:
            detecciones.append({"clase": nombre, "elemento": mapa["elemento"], "presente": mapa["presente"],
                                "confianza": round(float(conf), 3), "caja": [round(float(v), 1) for v in caja]})

    resultado_personas = []
    for p in personas:
        estado = {}
        for elem in requeridos:
            franja = tuple(zonas.get(elem, (0.0, 1.0)))
            if elem not in {c["elemento"] for c in clases.values()}:
                estado[elem] = {"estado": "no_soportado"}
                continue
            if not _zona_visible(p, franja, alto, ancho, int(_cfg.get("persona_altura_minima_px", 80))):
                estado[elem] = {"estado": "no_visible"}
                continue
            propias = [d for d in detecciones if d["elemento"] == elem and _solapamiento(d["caja"], p, franja) >= 0.5]
            positivas = [d for d in propias if d["presente"]]
            negativas = [d for d in propias if not d["presente"]]
            if positivas and (not negativas or max(x["confianza"] for x in positivas) >= max(x["confianza"] for x in negativas)):
                estado[elem] = {"estado": "presente", "confianza": max(x["confianza"] for x in positivas)}
            elif negativas:
                estado[elem] = {"estado": "falta", "confianza": max(x["confianza"] for x in negativas), "explicita": True}
            else:
                # La zona se ve y el elemento no aparece: falta inferida
                estado[elem] = {"estado": "falta", "confianza": None, "explicita": False}
        resultado_personas.append({"caja": [round(v, 1) for v in p], "elementos": estado})

    return {"disponible": True, "personas": resultado_personas, "detecciones": detecciones,
            "modelo": _cfg["nombre"], "apto_produccion": bool(_cfg.get("apto_produccion"))}


def resumen(analisis: dict, requeridos: list[str]) -> dict:
    """Faltantes y no visibles agregados para todo el frame."""
    faltantes, no_visibles, personas_sin = set(), set(), 0
    for p in analisis.get("personas", []):
        f = [e for e, v in p["elementos"].items() if v["estado"] == "falta"]
        faltantes.update(f)
        no_visibles.update(e for e, v in p["elementos"].items() if v["estado"] == "no_visible")
        personas_sin += 1 if f else 0
    return {"faltantes": sorted(faltantes), "no_visibles": sorted(no_visibles - faltantes),
            "personas": len(analisis.get("personas", [])), "personas_sin_epp": personas_sin}


def anotar(frame: np.ndarray, analisis: dict) -> np.ndarray:
    """Dibuja personas y estado por elemento (verde presente, rojo falta, gris no visible)."""
    img = frame.copy()
    colores = {"presente": (95, 160, 15), "falta": (48, 48, 197), "no_visible": (150, 140, 130)}
    for d in analisis.get("detecciones", []):
        x1, y1, x2, y2 = map(int, d["caja"])
        c = (95, 160, 15) if d["presente"] else (48, 48, 197)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)
    for i, p in enumerate(analisis.get("personas", []), 1):
        x1, y1, x2, y2 = map(int, p["caja"])
        hay_falta = any(v["estado"] == "falta" for v in p["elementos"].values())
        cv2.rectangle(img, (x1, y1), (x2, y2), (48, 48, 197) if hay_falta else (178, 111, 31), 2)
        y = max(18, y1 + 18)
        cv2.putText(img, f"#{i}", (x1 + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for elem, v in p["elementos"].items():
            y += 18
            txt = f"{elem}: {v['estado'].replace('_', ' ')}"
            cv2.putText(img, txt, (x1 + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (x1 + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colores.get(v["estado"], (255, 255, 255)), 1, cv2.LINE_AA)
    return img


# ── Uso en streams: suavizado temporal y envío a Laravel ─────

class SuavizadorEPP:
    """Una falta solo se confirma si aparece en 'minimo' de los últimos 'ventana' frames."""

    def __init__(self):
        cfg = (_cfg or {}).get("suavizado", {})
        self.ventana = int(cfg.get("ventana", 5))
        self.minimo = int(cfg.get("minimo", 3))
        self._hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.ventana))

    def actualizar(self, faltantes: list[str], requeridos: list[str]) -> list[str]:
        for elem in requeridos:
            self._hist[elem].append(elem in faltantes)
        return [e for e in requeridos if sum(self._hist[e]) >= self.minimo]


async def enviar_a_laravel(payload: dict) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Se leen al enviar: el .env se carga después de importar este módulo
            laravel = os.getenv("LARAVEL_URL", "http://localhost:8000")
            token = os.getenv("LARAVEL_SERVICE_TOKEN", "")
            resp = await client.post(f"{laravel}/api/epp/procesar", json=payload,
                                     headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
            return resp.status_code == 200
    except Exception as e:
        logger.error(f"[EPP] Error enviando a Laravel: {e}")
        return False


_ultima_captura: dict[int, float] = {}
_ultimo_diagnostico: dict[int, float] = {}
SEG_ENTRE_CAPTURAS = 30
SEG_ENTRE_DIAGNOSTICOS = 60


def procesar_frame_stream(frame: np.ndarray, camara_id: int, requeridos: list[str], suavizador: SuavizadorEPP) -> Optional[dict]:
    """Analiza un frame de stream. Devuelve el payload para Laravel o None si no hay nada que reportar."""
    import time
    analisis = analizar(frame, requeridos)
    if not analisis.get("disponible"):
        return None
    r = resumen(analisis, requeridos)
    confirmados = suavizador.actualizar(r["faltantes"], requeridos)
    ahora = time.monotonic()

    if not confirmados:
        # Solo diagnóstico de encuadre (zonas del cuerpo no visibles): como máximo 1 por minuto
        if not r["no_visibles"] or ahora - _ultimo_diagnostico.get(camara_id, 0) < SEG_ENTRE_DIAGNOSTICOS:
            return None
        _ultimo_diagnostico[camara_id] = ahora

    # Una captura cada SEG_ENTRE_CAPTURAS por cámara, para no llenar el disco
    captura = None
    if confirmados and ahora - _ultima_captura.get(camara_id, 0) >= SEG_ENTRE_CAPTURAS:
        captura = guardar_captura(anotar(frame, analisis), camara_id)
        _ultima_captura[camara_id] = ahora
    confs = [v["confianza"] for p in analisis["personas"] for e, v in p["elementos"].items()
             if e in confirmados and v.get("confianza")]
    return {
        "camara_id": camara_id,
        "faltantes": confirmados,
        "no_visibles": r["no_visibles"],
        "personas": r["personas"],
        "personas_sin_epp": r["personas_sin_epp"],
        "confianza": round(max(confs), 3) if confs else 0.5,
        "captura": captura,
    }
