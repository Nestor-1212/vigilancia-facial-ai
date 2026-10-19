"""
VigiFacial — Microservicio de Reconocimiento Facial
FastAPI + DeepFace + OpenCV

Endpoints:
  GET  /status                     — Health check
  POST /analizar                   — Analiza imagen subida (archivo)
  POST /analizar-url               — Analiza imagen desde URL
  POST /registrar-rostro/{id}      — Registra foto de referencia para persona
  DELETE /registrar-rostro/{id}    — Elimina fotos de referencia de persona
  POST /stream/start               — Inicia procesamiento de stream RTSP
  POST /stream/stop/{camara_id}    — Detiene un stream
  GET  /stream/list                — Lista streams activos
"""
import os
import io
import logging
import cv2
import numpy as np
import httpx

from pathlib import Path
from contextlib import asynccontextmanager

import asyncio
import base64
import re
from typing import Optional

from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from models.schemas import ResultadoAnalisis, ResultadoTapaboca, ResultadoCelular, StreamConfig, RegistrarRostroRequest
from services.detector import analizar_frame, registrar_rostro, FACES_DB
from services import ppe_detector, registro_modelos, descubrimiento
from services.mask_detector import analizar_frame_tapaboca
from services.phone_detector import analizar_frame_celular
from services.stream_worker import stream_manager
from services.frame_buffer import frame_buffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

LARAVEL = os.getenv("LARAVEL_URL", "http://localhost:8000")
TOKEN   = os.getenv("LARAVEL_SERVICE_TOKEN", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== VigiFacial IA — Iniciando ===")
    logger.info(f"  Laravel URL  : {LARAVEL}")
    logger.info(f"  Faces DB     : {FACES_DB}")
    logger.info(f"  Modelo       : {os.getenv('DEEPFACE_MODEL','Facenet512')}")
    Path(FACES_DB).mkdir(parents=True, exist_ok=True)
    Path("./capturas").mkdir(exist_ok=True)
    yield
    logger.info("=== VigiFacial IA — Deteniendo streams ===")
    stream_manager.stop_all()


app = FastAPI(
    title="VigiFacial — Microservicio IA",
    description="Reconocimiento facial con DeepFace + OpenCV para sistema de vigilancia",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────

@app.get("/status", tags=["Sistema"])
async def status():
    """Verifica que el servicio esté activo."""
    faces_db = Path(FACES_DB)
    personas_registradas = len([d for d in faces_db.iterdir() if d.is_dir()]) if faces_db.exists() else 0
    return {
        "status": "online",
        "modelo": os.getenv("DEEPFACE_MODEL", "Facenet512"),
        "detector": os.getenv("DEEPFACE_DETECTOR", "opencv"),
        "threshold": float(os.getenv("DEEPFACE_THRESHOLD", "0.40")),
        "personas_en_db": personas_registradas,
        "streams_activos": len(stream_manager.list_active()),
    }


# ──────────────────────────────────────────────────────────
# ANÁLISIS DE IMÁGENES
# ──────────────────────────────────────────────────────────

@app.post("/analizar", response_model=ResultadoAnalisis, tags=["Análisis"])
async def analizar_imagen(
    camara_id: int = Form(...),
    imagen: UploadFile = File(...),
):
    """
    Recibe una imagen (JPEG/PNG) y el ID de la cámara.
    Analiza rostros y devuelve resultado.
    """
    contenido = await imagen.read()
    frame = _bytes_a_frame(contenido)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    resultado = analizar_frame(frame, camara_id)
    return ResultadoAnalisis(**resultado)


@app.post("/analizar-url", response_model=ResultadoAnalisis, tags=["Análisis"])
async def analizar_desde_url(payload: dict):
    """
    Recibe { imagen_url, camara_id } — descarga la imagen y la analiza.
    Usado cuando Laravel envía la URL pública de una captura.
    """
    imagen_url = payload.get("imagen_url")
    camara_id  = payload.get("camara_id")

    if not imagen_url or not camara_id:
        raise HTTPException(status_code=422, detail="Falta imagen_url o camara_id")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(imagen_url)
            resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"No se pudo descargar la imagen: {e}")

    frame = _bytes_a_frame(resp.content)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen descargada")

    resultado = analizar_frame(frame, int(camara_id))
    return ResultadoAnalisis(**resultado)


# ──────────────────────────────────────────────────────────
# BASE DE DATOS DE ROSTROS
# ──────────────────────────────────────────────────────────

@app.post("/registrar-rostro/{persona_id}", tags=["Personas"])
async def registrar_rostro_endpoint(
    persona_id: int,
    foto: UploadFile = File(...),
):
    """
    Registra una foto de referencia para una persona.
    Se guarda en faces_db/persona_{id}/ para que DeepFace la indexe.
    """
    contenido = await foto.read()
    frame = _bytes_a_frame(contenido)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la foto")

    ruta = registrar_rostro(persona_id, frame)
    return {"mensaje": "Rostro registrado correctamente", "ruta": ruta, "persona_id": persona_id}


@app.delete("/registrar-rostro/{persona_id}", tags=["Personas"])
async def eliminar_rostros(persona_id: int):
    """Elimina todas las fotos de referencia de una persona."""
    import shutil
    carpeta = Path(FACES_DB) / f"persona_{persona_id}"
    if carpeta.exists():
        shutil.rmtree(carpeta)
        # Invalidar cache de DeepFace
        for pkl in Path(FACES_DB).glob("*.pkl"):
            pkl.unlink()
        return {"mensaje": f"Rostros de persona {persona_id} eliminados"}
    return {"mensaje": "No se encontraron fotos para esta persona"}


@app.get("/personas-registradas", tags=["Personas"])
async def personas_registradas():
    """Lista las personas con fotos en la base de datos local."""
    db = Path(FACES_DB)
    if not db.exists():
        return {"personas": []}
    personas = []
    for d in db.iterdir():
        if d.is_dir() and d.name.startswith("persona_"):
            fotos = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            personas.append({
                "persona_id": d.name.replace("persona_", ""),
                "fotos": len(fotos),
            })
    return {"personas": personas, "total": len(personas)}


# ──────────────────────────────────────────────────────────
# GESTIÓN DE STREAMS RTSP
# ──────────────────────────────────────────────────────────



@app.post("/stream/stop/{camara_id}", tags=["Streams"])
async def detener_stream(camara_id: int):
    """Detiene el procesamiento de un stream RTSP."""
    ok = stream_manager.stop(camara_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Stream no encontrado")
    return {"mensaje": f"Stream {camara_id} detenido"}


@app.get("/stream/list", tags=["Streams"])
async def listar_streams():
    """Lista los streams RTSP activos."""
    return {"streams": stream_manager.list_active(), "total": len(stream_manager.list_active())}


@app.post("/stream/stop-all", tags=["Streams"])
async def detener_todos():
    """Detiene todos los streams activos."""
    stream_manager.stop_all()
    return {"mensaje": "Todos los streams detenidos"}


@app.get("/stream/video/{camara_id}", tags=["Streams"])
async def video_mjpeg(camara_id: int):
    """
    Transmite el video de una cámara como MJPEG (~10 FPS).
    Úsalo en el browser como: <img src="http://localhost:8001/stream/video/1">
    Requiere que el stream esté activo (POST /stream/start).
    """
    _NO_SIGNAL = _crear_frame_sin_senal()

    async def generate():
        while True:
            frame = frame_buffer.get(camara_id) if stream_manager.is_running(camara_id) else None
            img = frame if frame is not None else _NO_SIGNAL
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


def _crear_frame_sin_senal() -> np.ndarray:
    """Frame negro con texto 'Sin señal' para cámaras sin stream activo."""
    f = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(f, "Sin senal", (220, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 2)
    cv2.putText(f, "Stream no iniciado", (185, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    return f


# ──────────────────────────────────────────────────────────
# DETECCIÓN DE TAPABOCA
# ──────────────────────────────────────────────────────────

@app.post("/tapaboca/analizar", response_model=ResultadoTapaboca, tags=["Tapaboca"])
async def analizar_tapaboca(
    camara_id: int = Form(...),
    imagen: UploadFile = File(...),
):
    """
    Recibe una imagen y detecta si los rostros visibles llevan tapaboca.
    Retorna cuántas personas NO llevan mascarilla.
    """
    contenido = await imagen.read()
    frame = _bytes_a_frame(contenido)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    resultado = analizar_frame_tapaboca(frame, camara_id)
    return ResultadoTapaboca(**{k: resultado[k] for k in ResultadoTapaboca.model_fields})


@app.post("/tapaboca/analizar-webcam", response_model=ResultadoTapaboca, tags=["Tapaboca"])
async def analizar_tapaboca_webcam(camara_id: int, indice_webcam: int = 0):
    """
    Captura un frame de la webcam local y lo analiza para tapaboca.
    Útil para pruebas sin stream continuo.
    """
    cap = cv2.VideoCapture(indice_webcam)
    if not cap.isOpened():
        raise HTTPException(status_code=503, detail=f"No se pudo abrir webcam {indice_webcam}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="No se obtuvo frame de la webcam")

    resultado = analizar_frame_tapaboca(frame, camara_id)
    return ResultadoTapaboca(**{k: resultado[k] for k in ResultadoTapaboca.model_fields})


# ──────────────────────────────────────────────────────────
# DETECCIÓN DE CELULAR EN MANO
# ──────────────────────────────────────────────────────────

@app.post("/celular/analizar", response_model=ResultadoCelular, tags=["Celular"])
async def analizar_celular(
    camara_id: int = Form(...),
    imagen: UploadFile = File(...),
):
    """
    Recibe una imagen y detecta si alguna persona tiene un celular en la mano.
    Usa YOLOv8 para detectar el dispositivo, MediaPipe para verificar la mano
    y DeepFace para identificar al portador.
    """
    contenido = await imagen.read()
    frame = _bytes_a_frame(contenido)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    resultado = analizar_frame_celular(frame, camara_id)
    data = {k: v for k, v in resultado.items() if k in ResultadoCelular.model_fields and v is not None}
    return ResultadoCelular(**data)


async def _fuente_desde_laravel(camara_id: int) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{LARAVEL}/api/camaras/{camara_id}/fuente",
                                 headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"})
            return r.json().get("url") if r.status_code == 200 else None
    except Exception:
        return None


@app.post("/stream/start", tags=["Streams"])
async def iniciar_stream(config: StreamConfig):
    """
    Inicia el procesamiento de un stream en background.
    - rtsp_url: URL RTSP o "webcam:0" para cámara local
    - modo: "reconocimiento_facial" | "tapaboca"
    """
    # La URL con credenciales se pide a Laravel con el token de servicio: la contraseña nunca pasa por el navegador
    fuente = await _fuente_desde_laravel(config.camara_id) or config.rtsp_url
    ok = stream_manager.start(config.camara_id, fuente, config.nombre, config.modo, config.requeridos)
    if not ok:
        raise HTTPException(status_code=409, detail="El stream ya está activo o se alcanzó el límite")
    return {
        "mensaje": f"Stream {config.camara_id} iniciado",
        "camara_id": config.camara_id,
        "modo": config.modo,
        "requeridos": config.requeridos,
    }


# ──────────────────────────────────────────────────────────
# EPP (Fase 2): capacidades del modelo y prueba con imágenes
# ──────────────────────────────────────────────────────────

@app.get("/capacidades", tags=["EPP"])
async def capacidades_epp():
    """Elementos de EPP que el modelo configurado sabe verificar."""
    return await asyncio.to_thread(ppe_detector.capacidades)


@app.post("/epp/probar", tags=["EPP"])
async def probar_epp(imagen: UploadFile = File(...), requeridos: str = Form("")):
    """
    Analiza una foto y devuelve, por persona, si cada elemento está presente, falta o no se ve,
    junto con la imagen anotada (JPEG en base64). No genera alertas.
    """
    datos = await imagen.read()
    frame = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="El archivo no es una imagen válida")
    lista = [r.strip() for r in requeridos.split(",") if r.strip()]
    if not lista:
        lista = (await asyncio.to_thread(ppe_detector.capacidades)).get("elementos", [])
    analisis = await asyncio.to_thread(ppe_detector.analizar, frame, lista)
    if not analisis.get("disponible"):
        raise HTTPException(status_code=503, detail=f"Modelo de EPP no disponible: {analisis.get('error')}")
    anotada = ppe_detector.anotar(frame, analisis)
    ok, jpg = cv2.imencode(".jpg", anotada, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {
        **analisis,
        "resumen": ppe_detector.resumen(analisis, lista),
        "requeridos": lista,
        "imagen_anotada": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else None,
        "ancho": int(frame.shape[1]), "alto": int(frame.shape[0]),
    }


# ──────────────────────────────────────────────────────────
# ALTA DE CÁMARAS: descubrimiento a partir de la IP
# ──────────────────────────────────────────────────────────

class DatosCamara(BaseModel):
    ip: str
    usuario: str = ""
    password: str = ""


@app.post("/camaras/descubrir", tags=["Cámaras"])
async def descubrir_camara(d: DatosCamara):
    """
    Encuentra el video de una cámara solo con IP, usuario y contraseña (ONVIF o rutas RTSP conocidas)
    y devuelve una vista previa. Las URLs se devuelven SIN credenciales.
    """
    ip = d.ip.strip()
    if not re.fullmatch(r"[A-Za-z0-9.\-]{1,253}", ip):
        raise HTTPException(status_code=422, detail="IP o nombre de equipo inválido")
    logger.info(f"[Descubrir] {ip} (usuario: {d.usuario or '—'})")
    return await asyncio.to_thread(descubrimiento.descubrir, ip, d.usuario, d.password)


# ──────────────────────────────────────────────────────────
# MODELOS Y ENTRENAMIENTO (Fase 3)
# ──────────────────────────────────────────────────────────

@app.get("/modelos", tags=["Modelos"])
async def listar_modelos():
    """Candidatos registrados (con métricas y aptitud), modelo activo y estado del conjunto de datos."""
    cfg = registro_modelos.config_activa()
    return {
        "activo": {"id": cfg.get("id"), "nombre": cfg.get("nombre"), "apto_produccion": bool(cfg.get("apto_produccion")),
                   "advertencia": cfg.get("advertencia"), "activado": cfg.get("activado")},
        "hay_anterior": (registro_modelos.DIR_MODELOS / "config.anterior.json").exists(),
        "candidatos": registro_modelos.listar_candidatos(),
        "dataset": registro_modelos.estado_dataset(),
        "criterios": {"map50": registro_modelos.UMBRAL_MAP50, "ap_por_clase": registro_modelos.UMBRAL_CLASE,
                      "ejemplos_por_clase": registro_modelos.IMAGENES_MIN_POR_CLASE, "val_min": registro_modelos.VAL_MIN},
    }


class ActivarModelo(BaseModel):
    forzar: bool = False


@app.post("/modelos/{candidato_id}/activar", tags=["Modelos"])
async def activar_modelo(candidato_id: str, datos: ActivarModelo):
    """Activa un candidato. Sin 'forzar', solo si cumple los criterios de producción."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidato_id):
        raise HTTPException(status_code=422, detail="Identificador inválido")
    try:
        r = registro_modelos.activar(candidato_id, datos.forzar)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    ok = await asyncio.to_thread(ppe_detector.recargar)
    if not ok:
        # Si el modelo nuevo no carga, se vuelve al anterior para no dejar el servicio sin EPP
        registro_modelos.volver_al_anterior()
        await asyncio.to_thread(ppe_detector.recargar)
        raise HTTPException(status_code=500, detail="El modelo no se pudo cargar; se restauró el anterior.")
    return r


@app.post("/modelos/anterior", tags=["Modelos"])
async def modelo_anterior():
    """Vuelve al modelo que estaba activo antes del último cambio."""
    try:
        r = registro_modelos.volver_al_anterior()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    await asyncio.to_thread(ppe_detector.recargar)
    return r


@app.post("/modelos/recargar", tags=["Modelos"])
async def recargar_modelo():
    """Recarga el modelo activo desde config.json (lo usa entrenamiento/activar.py)."""
    if not await asyncio.to_thread(ppe_detector.recargar):
        raise HTTPException(status_code=500, detail="No se pudo cargar el modelo activo")
    return {"id": registro_modelos.config_activa().get("id")}


class MuestraEntrenamiento(BaseModel):
    captura: str
    origen: str = "alerta"
    alerta_id: Optional[int] = None
    camara_id: Optional[int] = None
    validacion: Optional[str] = None
    faltantes: list[str] = []


@app.post("/dataset/muestras", tags=["Modelos"])
async def recibir_muestra(m: MuestraEntrenamiento):
    """Guarda una captura (ya almacenada por Laravel) como foto pendiente de etiquetar."""
    nombre = registro_modelos.guardar_muestra(m.captura, m.model_dump(exclude={"captura"}))
    if not nombre:
        raise HTTPException(status_code=422, detail="Captura no encontrada en el almacenamiento de Laravel")
    return {"guardada": nombre}


# ──────────────────────────────────────────────────────────
# SINCRONIZACIÓN CON LARAVEL
# ──────────────────────────────────────────────────────────

@app.post("/sincronizar-personas", tags=["Sistema"])
async def sincronizar_personas():
    """
    Descarga las fotos de referencia desde Laravel y las registra en la DB local.
    Llama a GET /api/personas y descarga cada foto_referencia.
    """
    if not TOKEN:
        raise HTTPException(status_code=401, detail="LARAVEL_SERVICE_TOKEN no configurado en .env")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{LARAVEL}/api/personas?activo=1&per_page=200",
                headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"},
            )
            resp.raise_for_status()
            personas = resp.json().get("data", [])

        registradas = 0
        for p in personas:
            if not p.get("foto_referencia"):
                continue
            foto_url = f"{LARAVEL}/storage/{p['foto_referencia']}"
            async with httpx.AsyncClient(timeout=15) as client:
                img_resp = await client.get(foto_url)
                if img_resp.status_code != 200:
                    continue
            frame = _bytes_a_frame(img_resp.content)
            if frame is not None:
                registrar_rostro(p["id"], frame)
                registradas += 1

        return {"mensaje": f"Sincronización completa — {registradas} personas registradas"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────

def _bytes_a_frame(data: bytes) -> np.ndarray | None:
    """Convierte bytes de imagen a ndarray de OpenCV (BGR)."""
    try:
        imagen = Image.open(io.BytesIO(data)).convert("RGB")
        frame  = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
        return frame
    except Exception:
        return None
