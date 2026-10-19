"""
Worker que lee streams RTSP/webcam con OpenCV y procesa frames en background.
Cada cámara corre en su propio hilo.

Fuentes soportadas:
  - URL RTSP  : "rtsp://user:pass@192.168.1.10:554/stream"
  - Webcam    : "webcam:0"  (índice de cámara local, 0 = principal)

Modos de detección:
  - "reconocimiento_facial"  — identifica personas contra la base de rostros
  - "tapaboca"               — detecta si los rostros llevan mascarilla
"""
import asyncio
import cv2
import logging
import os
from threading import Thread, Event
from typing import Dict, Literal, Optional

from services.detector import analizar_frame, enviar_resultado_a_laravel
from services.mask_detector import analizar_frame_tapaboca, debe_enviar_alerta
from services.helmet_detector import analizar_frame_casco, debe_enviar_alerta_casco
from services.phone_detector import analizar_frame_celular, enviar_celular_a_laravel, set_session_id
from services.frame_buffer import frame_buffer
from services.ppe_detector import SuavizadorEPP, procesar_frame_stream, enviar_a_laravel as enviar_epp_a_laravel

logger = logging.getLogger(__name__)

STREAM_FPS  = int(os.getenv("STREAM_FPS", "1"))
MAX_STREAMS = int(os.getenv("MAX_STREAMS", "10"))
SENAL_TIMEOUT = float(os.getenv("SENAL_TIMEOUT", str(max(10, 5 / STREAM_FPS))))

ModDeteccion = Literal[
    "reconocimiento_facial",
    "tapaboca", "casco", "ambos",
    "celular",
    "tapaboca_celular", "casco_celular", "ambos_celular",
]

# Mapa modo → conjunto de detectores activos
_MODOS_DETECTORES: dict[str, frozenset] = {
    "reconocimiento_facial": frozenset(),
    "tapaboca":              frozenset({"tapaboca"}),
    "casco":                 frozenset({"casco"}),
    "ambos":                 frozenset({"tapaboca", "casco"}),
    "celular":               frozenset({"celular"}),
    "tapaboca_celular":      frozenset({"tapaboca", "celular"}),
    "casco_celular":         frozenset({"casco", "celular"}),
    "ambos_celular":         frozenset({"tapaboca", "casco", "celular"}),
}


class StreamWorker:
    def __init__(
        self,
        camara_id: int,
        rtsp_url: str,
        nombre: str = "",
        modo: ModDeteccion = "reconocimiento_facial",
        requeridos: Optional[list[str]] = None,
    ):
        self.camara_id = camara_id
        self.rtsp_url  = rtsp_url
        self.nombre    = nombre
        self.modo      = modo
        # Fase 2: elementos que verifica el modelo de EPP (casco, botas, guantes…)
        self.requeridos = list(requeridos or [])
        self._suavizador = SuavizadorEPP() if self.requeridos else None
        self._stop_evt = Event()
        self._thread: Thread = None

    def start(self):
        self._stop_evt.clear()
        self._thread = Thread(
            target=self._run, daemon=True, name=f"stream-{self.camara_id}"
        )
        self._thread.start()
        logger.info(f"[Stream {self.camara_id}] Iniciado — {self.rtsp_url} (modo: {self.modo})")

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"[Stream {self.camara_id}] Detenido")

    # ── Fuente de video ──────────────────────────────────────
    def _abrir_captura(self) -> cv2.VideoCapture:
        """Abre VideoCapture para RTSP o para webcam local."""
        if self.rtsp_url.startswith("webcam:"):
            indice = int(self.rtsp_url.split(":")[1])
            logger.info(f"[Stream {self.camara_id}] Abriendo webcam índice {indice}")
            return cv2.VideoCapture(indice)
        return cv2.VideoCapture(self.rtsp_url)

    # ── Loop principal ────────────────────────────────────────
    def _run(self):
        cap = self._abrir_captura()
        if not cap.isOpened():
            logger.error(f"[Stream {self.camara_id}] No se pudo abrir: {self.rtsp_url}")
            return

        frame_interval = 1.0 / STREAM_FPS
        loop = asyncio.new_event_loop()

        try:
            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[Stream {self.camara_id}] Frame vacío — reintentando...")
                    cap.release()
                    cap = self._abrir_captura()
                    self._stop_evt.wait(3)
                    continue

                # Guardar frame para el endpoint MJPEG (live view en browser)
                frame_buffer.put(self.camara_id, frame)

                resultados = self._analizar(frame)
                self._enviar_si_relevante(loop, resultados)
                self._stop_evt.wait(frame_interval)
        finally:
            cap.release()
            loop.close()
            frame_buffer.remove(self.camara_id)

    # ── Análisis según modo ───────────────────────────────────
    def _analizar(self, frame) -> list[dict]:
        activos = _MODOS_DETECTORES.get(self.modo, frozenset())
        resultados = []
        if "tapaboca" in activos:
            resultados.append(analizar_frame_tapaboca(frame, self.camara_id))
        if "casco" in activos:
            resultados.append(analizar_frame_casco(frame, self.camara_id))
        if "celular" in activos:
            resultados.append(analizar_frame_celular(frame, self.camara_id))
        if self.requeridos:
            payload = procesar_frame_stream(frame, self.camara_id, self.requeridos, self._suavizador)
            if payload:
                resultados.append({"tipo_deteccion": "epp", **payload})
        if not resultados or all(r.get("tipo_deteccion") == "epp" for r in resultados):
            resultados.append(analizar_frame(frame, self.camara_id))
        return resultados

    # ── Envío a Laravel ───────────────────────────────────────
    def _enviar_si_relevante(self, loop: asyncio.AbstractEventLoop, resultados: list[dict]):
        for resultado in resultados:
            mensaje = resultado.get("mensaje", "")
            tipo    = resultado.get("tipo_deteccion", "reconocimiento_facial")

            if tipo == "tapaboca":
                if mensaje == "sin_tapaboca" and debe_enviar_alerta(self.camara_id):
                    loop.run_until_complete(enviar_resultado_a_laravel(resultado))
            elif tipo == "casco":
                if mensaje == "sin_casco" and debe_enviar_alerta_casco(self.camara_id):
                    loop.run_until_complete(enviar_resultado_a_laravel(resultado))
            elif tipo == "epp":
                datos = {k: v for k, v in resultado.items() if k != "tipo_deteccion"}
                loop.run_until_complete(enviar_epp_a_laravel(datos))
            elif tipo == "celular":
                if resultado.get("es_nuevo"):
                    session_id = loop.run_until_complete(enviar_celular_a_laravel(resultado))
                    if session_id:
                        set_session_id(self.camara_id, session_id)
                elif resultado.get("finalizado"):
                    loop.run_until_complete(enviar_celular_a_laravel(resultado))
            else:
                if mensaje != "sin_rostro":
                    loop.run_until_complete(enviar_resultado_a_laravel(resultado))


class StreamManager:
    """Gestiona todos los workers activos."""

    def __init__(self):
        self._workers: Dict[int, StreamWorker] = {}

    def start(
        self,
        camara_id: int,
        rtsp_url: str,
        nombre: str = "",
        modo: ModDeteccion = "reconocimiento_facial",
        requeridos: Optional[list[str]] = None,
    ) -> bool:
        if camara_id in self._workers:
            logger.info(f"[Stream {camara_id}] Ya está corriendo")
            return False
        if len(self._workers) >= MAX_STREAMS:
            logger.warning("Límite de streams alcanzado")
            return False

        worker = StreamWorker(camara_id, rtsp_url, nombre, modo, requeridos)
        worker.start()
        self._workers[camara_id] = worker
        return True

    def stop(self, camara_id: int) -> bool:
        worker = self._workers.pop(camara_id, None)
        if not worker:
            return False
        worker.stop()
        return True

    def stop_all(self):
        for worker in list(self._workers.values()):
            worker.stop()
        self._workers.clear()

    def list_active(self) -> list:
        activos = []
        for cid, w in self._workers.items():
            edad = frame_buffer.segundos_desde_ultimo(cid)
            activos.append({
                "camara_id": cid,
                "nombre": w.nombre,
                "url": w.rtsp_url,
                "modo": w.modo,
                "requeridos": w.requeridos,
                # Hay señal si llegó un frame en los últimos SENAL_TIMEOUT segundos
                "con_senal": edad is not None and edad <= SENAL_TIMEOUT,
                "segundos_sin_frame": None if edad is None else round(edad, 1),
            })
        return activos

    def is_running(self, camara_id: int) -> bool:
        return camara_id in self._workers


# Instancia global
stream_manager = StreamManager()
