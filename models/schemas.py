from pydantic import BaseModel
from typing import Optional, Literal


class ResultadoAnalisis(BaseModel):
    camara_id: int
    identificado: bool
    persona_id: Optional[int] = None
    confianza: float
    captura: Optional[str] = None
    embedding: Optional[list[float]] = None
    mensaje: str = ""


class ResultadoTapaboca(BaseModel):
    camara_id: int
    mensaje: str
    total_rostros: int
    sin_tapaboca: int
    confianza: float
    tipo_deteccion: str = "tapaboca"


class StreamConfig(BaseModel):
    camara_id: int
    rtsp_url: str
    nombre: str = ""
    modo: Literal["reconocimiento_facial", "tapaboca", "casco", "ambos"] = "reconocimiento_facial"


class RegistrarRostroRequest(BaseModel):
    persona_id: int
    nombre: str
