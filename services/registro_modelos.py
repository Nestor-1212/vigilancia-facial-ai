"""
Registro de modelos de EPP y del conjunto de datos de entrenamiento (Fase 3).

Estructura:
  modelos_epp/
    config.json                 modelo ACTIVO (lo escribe activar())
    config.anterior.json        respaldo para volver atrás
    candidatos/<id>/
      modelo.pt                 pesos
      clases.json               clase del modelo → {elemento, presente}
      metricas.json             mAP general y por clase, origen, fecha…
  entrenamiento/dataset/
    clases.json                 clases a entrenar (mismo formato que arriba, en orden)
    data.yaml                   lo genera preparar.py para Ultralytics
    pendientes/                 fotos sin etiquetar (+ .json con su origen)
    images/{train,val}/  labels/{train,val}/   formato YOLO
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ / "modelos_epp"
DIR_CANDIDATOS = DIR_MODELOS / "candidatos"
DIR_DATASET = RAIZ / "entrenamiento" / "dataset"
EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Criterios mínimos para considerar un modelo apto para producción
UMBRAL_MAP50 = 0.70          # mAP@0.5 general
UMBRAL_CLASE = 0.60          # AP@0.5 de cada clase de presencia
IMAGENES_MIN_POR_CLASE = 200 # instancias etiquetadas por clase de presencia
VAL_MIN = 0.15               # fracción mínima de imágenes en validación


def _leer(ruta: Path, defecto=None):
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return defecto


def _escribir(ruta: Path, datos) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def config_activa() -> dict:
    return _leer(DIR_MODELOS / "config.json", {})


# ── Aptitud para producción ──────────────────────────────────

def evaluar_aptitud(metricas: dict, clases: dict) -> tuple[bool, list[str]]:
    """Devuelve (apto, motivos por los que no lo es)."""
    motivos = []
    if metricas.get("origen") == "externo" and not metricas.get("licencia_comercial"):
        motivos.append("Modelo externo sin licencia comercial verificada.")
    map50 = metricas.get("map50")
    if map50 is None:
        motivos.append("Sin métricas medidas.")
    elif map50 < UMBRAL_MAP50:
        motivos.append(f"mAP@0.5 = {map50:.2f}, por debajo de {UMBRAL_MAP50:.2f}.")
    por_clase = metricas.get("por_clase") or {}
    for nombre, mapa in clases.items():
        if mapa.get("presente") and nombre in por_clase and por_clase[nombre] < UMBRAL_CLASE:
            motivos.append(f"Clase «{nombre}» con AP@0.5 = {por_clase[nombre]:.2f} (mínimo {UMBRAL_CLASE:.2f}).")
    if metricas.get("imagenes_val") is not None and metricas["imagenes_val"] < 50:
        n = metricas["imagenes_val"]
        motivos.append(f"Evaluado con solo {n} {'imagen' if n == 1 else 'imágenes'} de validación (mínimo recomendado 50).")
    return (not motivos, motivos)


# ── Candidatos ───────────────────────────────────────────────

def listar_candidatos() -> list[dict]:
    activo = config_activa().get("id")
    salida = []
    if not DIR_CANDIDATOS.exists():
        return salida
    for d in sorted(DIR_CANDIDATOS.iterdir()):
        if not d.is_dir():
            continue
        metricas = _leer(d / "metricas.json", {}) or {}
        clases = _leer(d / "clases.json", {}) or {}
        apto, motivos = evaluar_aptitud(metricas, clases)
        salida.append({
            "id": d.name,
            "nombre": metricas.get("nombre", d.name),
            "origen": metricas.get("origen", "entrenado"),
            "creado": metricas.get("creado"),
            "base": metricas.get("base"),
            "map50": metricas.get("map50"),
            "map50_95": metricas.get("map50_95"),
            "por_clase": metricas.get("por_clase") or {},
            "imagenes_entrenamiento": metricas.get("imagenes_train"),
            "imagenes_val": metricas.get("imagenes_val"),
            "elementos": sorted({c["elemento"] for c in clases.values()}),
            "clases": list(clases.keys()),
            "apto_produccion": apto,
            "motivos": motivos,
            "notas": metricas.get("notas"),
            "tiene_pesos": (d / "modelo.pt").exists(),
            "activo": d.name == activo,
        })
    # El activo arriba; el resto, los más recientes primero
    salida.sort(key=lambda c: c["creado"] or "", reverse=True)
    salida.sort(key=lambda c: not c["activo"])
    return salida


def activar(candidato_id: str, forzar: bool = False) -> dict:
    """Activa un candidato. Sin 'forzar', solo si cumple los criterios de producción."""
    d = DIR_CANDIDATOS / candidato_id
    if not (d / "modelo.pt").exists():
        raise FileNotFoundError(f"El candidato «{candidato_id}» no tiene pesos (modelo.pt).")
    metricas = _leer(d / "metricas.json", {}) or {}
    clases = _leer(d / "clases.json", {}) or {}
    if not clases:
        raise ValueError(f"El candidato «{candidato_id}» no tiene clases.json.")
    apto, motivos = evaluar_aptitud(metricas, clases)
    if not apto and not forzar:
        raise PermissionError("No cumple los criterios de producción: " + " ".join(motivos))

    actual = config_activa()
    if actual:
        _escribir(DIR_MODELOS / "config.anterior.json", actual)

    nueva = {**actual}
    nueva.update({
        "id": candidato_id,
        "archivo": f"candidatos/{candidato_id}/modelo.pt",
        "nombre": metricas.get("nombre", candidato_id),
        "apto_produccion": apto,
        "advertencia": None if apto else ("Activado sin cumplir los criterios de producción: " + " ".join(motivos)),
        "clases": clases,
        "activado": datetime.now().isoformat(timespec="seconds"),
    })
    nueva.setdefault("confianza_minima", 0.25)
    _escribir(DIR_MODELOS / "config.json", nueva)
    return {"id": candidato_id, "apto_produccion": apto, "motivos": motivos}


def volver_al_anterior() -> dict:
    anterior = _leer(DIR_MODELOS / "config.anterior.json")
    if not anterior:
        raise FileNotFoundError("No hay un modelo anterior guardado.")
    actual = config_activa()
    _escribir(DIR_MODELOS / "config.json", anterior)
    _escribir(DIR_MODELOS / "config.anterior.json", actual)
    return {"id": anterior.get("id")}


# ── Conjunto de datos ────────────────────────────────────────

def clases_dataset() -> dict:
    return _leer(DIR_DATASET / "clases.json", {}) or {}


def _imagenes(carpeta: Path) -> list[Path]:
    return [p for p in carpeta.glob("*") if p.suffix.lower() in EXT_IMAGEN] if carpeta.exists() else []


def estado_dataset() -> dict:
    clases = clases_dataset()
    nombres = list(clases.keys())
    instancias = {n: 0 for n in nombres}
    instancias_val = 0
    conteo = {}
    sin_etiqueta = 0
    for parte in ("train", "val"):
        imgs = _imagenes(DIR_DATASET / "images" / parte)
        conteo[parte] = len(imgs)
        for img in imgs:
            lbl = DIR_DATASET / "labels" / parte / (img.stem + ".txt")
            if not lbl.exists():
                sin_etiqueta += 1
                continue
            for linea in lbl.read_text(encoding="utf-8").splitlines():
                partes = linea.split()
                if partes and partes[0].isdigit() and int(partes[0]) < len(nombres):
                    instancias[nombres[int(partes[0])]] += 1
                    instancias_val += parte == "val"

    pendientes = _imagenes(DIR_DATASET / "pendientes")
    preanotadas = sum(1 for p in pendientes if p.with_suffix(".txt").exists())
    por_origen = {}
    for p in pendientes:
        origen = (_leer(p.with_suffix(".json"), {}) or {}).get("origen", "importada")
        por_origen[origen] = por_origen.get(origen, 0) + 1

    total = conteo.get("train", 0) + conteo.get("val", 0)
    presencia = [n for n, m in clases.items() if m.get("presente")]
    requisitos = [
        {"texto": "Clases definidas en entrenamiento/dataset/clases.json", "ok": bool(clases)},
        {"texto": f"Al menos {IMAGENES_MIN_POR_CLASE} ejemplos etiquetados por cada elemento",
         "ok": bool(presencia) and all(instancias.get(n, 0) >= IMAGENES_MIN_POR_CLASE for n in presencia)},
        {"texto": f"Al menos {int(VAL_MIN * 100)}% de las imágenes en validación",
         "ok": total > 0 and conteo.get("val", 0) / total >= VAL_MIN},
        {"texto": "Todas las imágenes de train/val tienen su etiqueta", "ok": total > 0 and sin_etiqueta == 0},
    ]
    return {
        "clases": nombres,
        "elementos": {n: clases[n] for n in nombres},
        "imagenes": {"train": conteo.get("train", 0), "val": conteo.get("val", 0), "sin_etiqueta": sin_etiqueta},
        "instancias": instancias,
        "instancias_val": instancias_val,
        "pendientes": {"total": len(pendientes), "preanotadas": preanotadas, "por_origen": por_origen},
        "requisitos": requisitos,
        "listo": all(r["ok"] for r in requisitos),
        "minimo_por_clase": IMAGENES_MIN_POR_CLASE,
    }


def guardar_muestra(ruta_captura: str, metadatos: dict) -> Optional[str]:
    """Copia una captura del almacenamiento de Laravel a dataset/pendientes con su origen."""
    raiz_laravel = Path(os.getenv("LARAVEL_STORAGE_PATH", ""))
    origen = (raiz_laravel / ruta_captura).resolve()
    # Solo se aceptan rutas dentro del almacenamiento público de Laravel
    if not str(origen).startswith(str(raiz_laravel.resolve())) or not origen.exists() or origen.suffix.lower() not in EXT_IMAGEN:
        return None
    destino_dir = DIR_DATASET / "pendientes"
    destino_dir.mkdir(parents=True, exist_ok=True)
    nombre = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{origen.name}"
    shutil.copy2(origen, destino_dir / nombre)
    _escribir((destino_dir / nombre).with_suffix(".json"), {**metadatos, "recibida": datetime.now().isoformat(timespec="seconds")})
    return nombre
