"""
Buffer thread-safe de último frame por cámara.
El stream_worker escribe aquí; el endpoint MJPEG lee de aquí.
Así ambos comparten el mismo capture sin conflicto.
"""
import threading
import numpy as np
from typing import Optional


class FrameBuffer:
    def __init__(self):
        self._frames: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def put(self, camara_id: int, frame: np.ndarray):
        with self._lock:
            self._frames[camara_id] = frame.copy()

    def get(self, camara_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._frames.get(camara_id)

    def remove(self, camara_id: int):
        with self._lock:
            self._frames.pop(camara_id, None)

    def active_ids(self) -> list[int]:
        with self._lock:
            return list(self._frames.keys())


frame_buffer = FrameBuffer()
