"""
Entrena un modelo de EPP con el conjunto de datos y lo registra como CANDIDATO
(no lo activa: eso se hace desde Configuración → Modelos o con activar.py).

  venv\\python.exe entrenamiento\\entrenar.py --id cliente_x_v1 [--epocas 100] [--imgsz 640]
                                             [--base activo|yolov8s.pt|ruta.pt] [--dispositivo cpu|0] [--lote 16]

Recomendado: GPU (--dispositivo 0). En CPU funciona, pero un entrenamiento real puede tardar horas.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.registro_modelos import (  # noqa: E402
    DIR_CANDIDATOS, DIR_DATASET, DIR_MODELOS, clases_dataset, config_activa, estado_dataset, evaluar_aptitud,
)

DIR_EJECUCIONES = Path(__file__).resolve().parent / "ejecuciones"


def metricas_de_validacion(modelo, data_yaml: Path, imgsz: int, dispositivo: str) -> dict:
    """mAP general y AP@0.5 por clase sobre el conjunto de validación."""
    v = modelo.val(data=str(data_yaml), imgsz=imgsz, device=dispositivo, plots=False, verbose=False)
    # Sin etiquetas en validación, Ultralytics devuelve listas vacías en lugar de arreglos
    lista = lambda x: x.tolist() if hasattr(x, "tolist") else list(x)  # noqa: E731
    por_clase = {}
    for idx, ap50 in zip(lista(v.box.ap_class_index), lista(v.box.ap50)):
        por_clase[modelo.names[int(idx)]] = round(float(ap50), 4)
    return {"map50": round(float(v.box.map50), 4), "map50_95": round(float(v.box.map), 4), "por_clase": por_clase}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default=f"entrenado_{datetime.now():%Y%m%d_%H%M}", help="identificador del candidato (sin espacios)")
    ap.add_argument("--nombre", help="nombre legible del modelo")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--base", default="activo", help="'activo' = afinar el modelo activo; o un .pt (p. ej. yolov8s.pt)")
    ap.add_argument("--dispositivo", default="cpu")
    a = ap.parse_args()

    if not a.id.replace("_", "").replace("-", "").isalnum():
        sys.exit("El --id solo puede tener letras, números, guiones y guiones bajos.")
    destino = DIR_CANDIDATOS / a.id
    if destino.exists():
        sys.exit(f"Ya existe el candidato «{a.id}». Usa otro --id.")
    data_yaml = DIR_DATASET / "data.yaml"
    if not data_yaml.exists():
        sys.exit("Falta dataset/data.yaml. Ejecuta: preparar.py validar")
    estado = estado_dataset()
    if not estado["imagenes"]["train"] or not estado["imagenes"]["val"]:
        sys.exit("El dataset necesita imágenes etiquetadas en train y en val (preparar.py dividir).")
    if not estado["listo"]:
        print("AVISO: el dataset no cumple los requisitos de producción; el modelo resultante será solo de prueba.")
    if not estado.get("instancias_val"):
        print("AVISO: las imágenes de validación no tienen ningún objeto etiquetado: las métricas saldrán en 0.")

    base = str(DIR_MODELOS / config_activa()["archivo"]) if a.base == "activo" else a.base
    print(f"Base: {base}\nDataset: {estado['imagenes']['train']} train · {estado['imagenes']['val']} val\n")

    from ultralytics import YOLO
    modelo = YOLO(base)
    resultado = modelo.train(data=str(data_yaml), epochs=a.epocas, imgsz=a.imgsz, batch=a.lote, device=a.dispositivo,
                             project=str(DIR_EJECUCIONES), name=a.id, exist_ok=True, plots=False, verbose=False)
    mejor = Path(resultado.save_dir) / "weights" / "best.pt"
    if not mejor.exists():
        sys.exit(f"El entrenamiento no produjo {mejor}")

    entrenado = YOLO(str(mejor))
    metricas = metricas_de_validacion(entrenado, data_yaml, a.imgsz, a.dispositivo)
    clases = clases_dataset()

    destino.mkdir(parents=True)
    shutil.copy2(mejor, destino / "modelo.pt")
    (destino / "clases.json").write_text(json.dumps(clases, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "id": a.id,
        "nombre": a.nombre or f"Modelo entrenado {a.id}",
        "origen": "entrenado",
        "creado": datetime.now().isoformat(timespec="seconds"),
        "base": Path(base).name if a.base != "activo" else f"activo ({config_activa().get('id')})",
        "epocas": a.epocas, "imgsz": a.imgsz,
        "imagenes_train": estado["imagenes"]["train"], "imagenes_val": estado["imagenes"]["val"],
        **metricas,
        "notas": "Métricas medidas con el conjunto de validación propio.",
    }
    apto, motivos = evaluar_aptitud(info, clases)
    info["apto_produccion"] = apto
    (destino / "metricas.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nCandidato «{a.id}» registrado · mAP@0.5 = {metricas['map50']:.3f}")
    for clase, ap50 in metricas["por_clase"].items():
        print(f"  {clase:<20} AP@0.5 = {ap50:.3f}")
    print("\nApto para producción." if apto else "\nNo apto para producción:\n  - " + "\n  - ".join(motivos))
    print("Actívalo desde Configuración → Modelos o con: entrenamiento\\activar.py", a.id)


if __name__ == "__main__":
    main()
