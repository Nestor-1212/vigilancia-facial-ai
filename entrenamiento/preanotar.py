"""
Preanota las fotos de dataset/pendientes con el modelo activo, para que las personas
solo corrijan en lugar de dibujar todo desde cero.

  venv\\python.exe entrenamiento\\preanotar.py [--confianza 0.25] [--sobrescribir]

Escribe un .txt YOLO junto a cada foto (solo con las clases de dataset/clases.json que el
modelo activo conoce, por nombre) y marca la foto como "preanotada" en su .json.
Después de revisarlas en Label Studio, CVAT o Roboflow, marca "revisada": true.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.registro_modelos import DIR_DATASET, DIR_MODELOS, EXT_IMAGEN, clases_dataset, config_activa  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confianza", type=float, default=0.25)
    ap.add_argument("--sobrescribir", action="store_true", help="rehacer también las que ya tienen .txt")
    a = ap.parse_args()

    clases = list(clases_dataset().keys())
    if not clases:
        sys.exit("Falta dataset/clases.json. Ejecuta: preparar.py iniciar")
    cfg = config_activa()
    from ultralytics import YOLO
    modelo = YOLO(str(DIR_MODELOS / cfg["archivo"]))
    # Clase del modelo → id en el dataset (por nombre)
    a_dataset = {i: clases.index(n) for i, n in modelo.names.items() if n in clases}
    if not a_dataset:
        sys.exit("El modelo activo no comparte ninguna clase con dataset/clases.json: no hay nada que preanotar.")
    print(f"Modelo activo: {cfg.get('nombre')} · clases en común: {', '.join(modelo.names[i] for i in a_dataset)}")

    pend = DIR_DATASET / "pendientes"
    fotos = [p for p in sorted(pend.glob("*")) if p.suffix.lower() in EXT_IMAGEN and (a.sobrescribir or not p.with_suffix(".txt").exists())]
    for n, img in enumerate(fotos, 1):
        r = modelo.predict(str(img), conf=a.confianza, verbose=False)[0]
        lineas = []
        for (x, y, w, h), cls in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist()):
            if int(cls) in a_dataset:
                lineas.append(f"{a_dataset[int(cls)]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        img.with_suffix(".txt").write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
        meta_p = img.with_suffix(".json")
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {"origen": "importada"}
        meta.update({"preanotada": True, "revisada": False, "modelo_preanotacion": cfg.get("id")})
        meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(f"  [{n}/{len(fotos)}] {img.name}: {len(lineas)} objetos")
    print(f"\nPreanotadas {len(fotos)} fotos. Revísalas y corrígelas antes de usar preparar.py dividir.")


if __name__ == "__main__":
    main()
