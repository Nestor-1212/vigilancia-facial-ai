"""
Buffer thread-safe de último frame por cámara.
El stream_worker escribe aquí; el endpoint MJPEG lee de aquí.
Así ambos comparten el mismo capture sin conflicto.
"""
import threading
import time
import numpy as np
from typing import Optional


class FrameBuffer:
    def __init__(self):
        self._frames: dict[int, np.ndarray] = {}
        self._ts: dict[int, float] = {}
        self._lock = threading.Lock()

    def put(self, camara_id: int, frame: np.ndarray):
        with self._lock:
            self._frames[camara_id] = frame.copy()
            self._ts[camara_id] = time.monotonic()

    def get(self, camara_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._frames.get(camara_id)

    def segundos_desde_ultimo(self, camara_id: int) -> Optional[float]:
        """Segundos desde el último frame recibido, o None si nunca llegó uno."""
        with self._lock:
            ts = self._ts.get(camara_id)
        return None if ts is None else time.monotonic() - ts

    def remove(self, camara_id: int):
        with self._lock:
            self._frames.pop(camara_id, None)
            self._ts.pop(camara_id, None)

    def active_ids(self) -> list[int]:
        with self._lock:
            return list(self._frames.keys())


frame_buffer = FrameBuffer()
