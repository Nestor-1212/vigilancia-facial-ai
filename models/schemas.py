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


class ResultadoCelular(BaseModel):
    camara_id: int
    mensaje: str
    tipo_deteccion: str = "celular"
    identificado: bool = False
    persona_id: Optional[int] = None
    confianza: float = 0.0
    confianza_facial: Optional[float] = None
    foto_rostro: Optional[str] = None
    inicio: Optional[str] = None
    fin: Optional[str] = None
    duracion_segundos: Optional[int] = None
    session_id: Optional[int] = None
    es_nuevo: bool = False
    finalizado: bool = False


class StreamConfig(BaseModel):
    camara_id: int
    rtsp_url: str
    nombre: str = ""
    modo: Literal[
        "reconocimiento_facial",
        "tapaboca", "casco", "ambos",
        "celular",
        "tapaboca_celular", "casco_celular", "ambos_celular",
    ] = "reconocimiento_facial"


class RegistrarRostroRequest(BaseModel):
    persona_id: int
    nombre: str
