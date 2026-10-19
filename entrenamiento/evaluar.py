"""
Mide uno o varios candidatos con el conjunto de VALIDACIÓN propio y actualiza sus métricas.
Sirve para comparar modelos con las mismas fotos (incluido el de referencia).

  venv\\python.exe entrenamiento\\evaluar.py [--id referencia] [--dispositivo cpu] [--imgsz 640]

Solo se pueden evaluar candidatos cuyas clases coinciden, en nombre y orden, con dataset/clases.json.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.registro_modelos import DIR_CANDIDATOS, DIR_DATASET, clases_dataset, estado_dataset, evaluar_aptitud  # noqa: E402
from entrenamiento.entrenar import metricas_de_validacion  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", help="candidato a evaluar (por defecto, todos los compatibles)")
    ap.add_argument("--dispositivo", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    data_yaml = DIR_DATASET / "data.yaml"
    if not data_yaml.exists():
        sys.exit("Falta dataset/data.yaml. Ejecuta: preparar.py validar")
    n_val = estado_dataset()["imagenes"]["val"]
    if not n_val:
        sys.exit("No hay imágenes de validación.")
    nombres = list(clases_dataset().keys())

    from ultralytics import YOLO
    ids = [a.id] if a.id else [d.name for d in sorted(DIR_CANDIDATOS.iterdir()) if (d / "modelo.pt").exists()]
    for cid in ids:
        d = DIR_CANDIDATOS / cid
        modelo = YOLO(str(d / "modelo.pt"))
        if list(modelo.names.values()) != nombres:
            print(f"  {cid}: omitido (sus clases no coinciden con dataset/clases.json)")
            continue
        m = metricas_de_validacion(modelo, data_yaml, a.imgsz, a.dispositivo)
        info = json.loads((d / "metricas.json").read_text(encoding="utf-8")) if (d / "metricas.json").exists() else {"id": cid}
        info.update(m)
        info.update({"imagenes_val": n_val, "evaluado": datetime.now().isoformat(timespec="seconds"),
                     "notas": "Métricas medidas con el conjunto de validación propio."})
        clases = json.loads((d / "clases.json").read_text(encoding="utf-8"))
        info["apto_produccion"], _ = evaluar_aptitud(info, clases)
        (d / "metricas.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {cid}: mAP@0.5 = {m['map50']:.3f} con {n_val} imágenes de validación")


if __name__ == "__main__":
    main()
