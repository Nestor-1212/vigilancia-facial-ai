"""
Prepara el conjunto de datos para entrenar un modelo de EPP.

  venv\\python.exe entrenamiento\\preparar.py iniciar            crea la estructura y clases.json (partiendo del modelo activo)
  venv\\python.exe entrenamiento\\preparar.py importar <carpeta> copia fotos a dataset/pendientes
  venv\\python.exe entrenamiento\\preparar.py dividir [--val 0.2] mueve las fotos YA etiquetadas a train/val
  venv\\python.exe entrenamiento\\preparar.py validar            revisa etiquetas, genera data.yaml y muestra el estado

Las etiquetas usan el formato YOLO: un .txt por imagen, con el mismo nombre,
y una línea por objeto: <id_clase> <x_centro> <y_centro> <ancho> <alto> (valores 0–1).
"""
import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.registro_modelos import (  # noqa: E402
    DIR_DATASET, DIR_MODELOS, EXT_IMAGEN, clases_dataset, config_activa, estado_dataset,
)


def iniciar(_args):
    for sub in ("pendientes", "images/train", "images/val", "labels/train", "labels/val"):
        (DIR_DATASET / sub).mkdir(parents=True, exist_ok=True)
    ruta = DIR_DATASET / "clases.json"
    if ruta.exists():
        print(f"Ya existe {ruta.relative_to(DIR_DATASET.parent.parent)}; no se modifica.")
    else:
        # Se parte de las clases del modelo activo, EN SU MISMO ORDEN (así evaluar.py puede medir el
        # modelo activo con las fotos del cliente). Agrega al final las nuevas (p. ej. el uniforme).
        cfg = config_activa()
        mapa = cfg.get("clases", {})
        try:
            from ultralytics import YOLO
            orden = list(YOLO(str(DIR_MODELOS / cfg["archivo"])).names.values())
        except Exception:
            orden = list(mapa.keys())
        clases = {n: mapa[n] for n in orden if n in mapa}
        ruta.write_text(json.dumps(clases, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Creado clases.json con {len(clases)} clases del modelo activo. Edítalo para agregar o quitar clases.")
    print("Estructura lista en", DIR_DATASET)


def importar(args):
    origen = Path(args.carpeta)
    if not origen.is_dir():
        sys.exit(f"No existe la carpeta {origen}")
    destino = DIR_DATASET / "pendientes"
    destino.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in origen.rglob("*"):
        if img.suffix.lower() not in EXT_IMAGEN:
            continue
        nombre = f"imp_{datetime.now():%Y%m%d}_{n:05d}{img.suffix.lower()}"
        shutil.copy2(img, destino / nombre)
        # Si la foto ya trae su etiqueta YOLO al lado, también se copia
        lbl = img.with_suffix(".txt")
        if lbl.exists():
            shutil.copy2(lbl, (destino / nombre).with_suffix(".txt"))
        (destino / nombre).with_suffix(".json").write_text(json.dumps(
            {"origen": "importada", "archivo_original": str(img)}, ensure_ascii=False), encoding="utf-8")
        n += 1
    print(f"Importadas {n} fotos a {destino}")


def _meta(img: Path) -> dict:
    try:
        return json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _revisada(img: Path) -> bool:
    """Una etiqueta hecha por una persona, o una preanotada que alguien marcó como revisada."""
    m = _meta(img)
    return not m.get("preanotada") or bool(m.get("revisada"))


def dividir(args):
    pend = DIR_DATASET / "pendientes"
    etiquetadas = [p for p in pend.glob("*") if p.suffix.lower() in EXT_IMAGEN and p.with_suffix(".txt").exists()]
    # Las preanotadas sin revisar no se usan: primero hay que corregirlas
    revisadas = etiquetadas if args.incluir_preanotadas else [p for p in etiquetadas if _revisada(p)]
    random.seed(42)
    random.shuffle(revisadas)
    n_val = max(1, round(len(revisadas) * args.val)) if revisadas else 0
    for i, img in enumerate(revisadas):
        parte = "val" if i < n_val else "train"
        shutil.move(str(img), DIR_DATASET / "images" / parte / img.name)
        shutil.move(str(img.with_suffix(".txt")), DIR_DATASET / "labels" / parte / (img.stem + ".txt"))
        meta = img.with_suffix(".json")
        if meta.exists():
            meta.unlink()
    print(f"Movidas {len(revisadas)} fotos etiquetadas: {len(revisadas) - n_val} a train, {n_val} a val.")
    sin = len(etiquetadas) - len(revisadas)
    if sin:
        print(f"{sin} fotos preanotadas siguen en pendientes: revisa sus etiquetas y marca \"revisada\": true en su .json "
              "(o usa --incluir-preanotadas bajo tu responsabilidad).")


def validar(_args):
    clases = clases_dataset()
    if not clases:
        sys.exit("Falta entrenamiento/dataset/clases.json. Ejecuta primero: preparar.py iniciar")
    nombres = list(clases.keys())
    errores = 0
    for parte in ("train", "val"):
        for lbl in (DIR_DATASET / "labels" / parte).glob("*.txt"):
            for i, linea in enumerate(lbl.read_text(encoding="utf-8").splitlines(), 1):
                p = linea.split()
                if not p:
                    continue
                ok = len(p) == 5 and p[0].isdigit() and int(p[0]) < len(nombres) and all(0 <= float(v) <= 1 for v in p[1:])
                if not ok:
                    errores += 1
                    print(f"  Etiqueta inválida: {lbl.name} línea {i}: {linea}")
    yaml = [f"path: {DIR_DATASET.as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(nombres)]
    (DIR_DATASET / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    e = estado_dataset()
    print(f"\nClases: {', '.join(nombres)}")
    print(f"Imágenes: {e['imagenes']['train']} train · {e['imagenes']['val']} val · {e['imagenes']['sin_etiqueta']} sin etiqueta")
    print(f"Pendientes: {e['pendientes']['total']} ({e['pendientes']['preanotadas']} preanotadas)")
    print("Ejemplos por clase:", ", ".join(f"{k}={v}" for k, v in e["instancias"].items()))
    print("\nRequisitos para entrenar un modelo de producción:")
    for r in e["requisitos"]:
        print(f"  [{'x' if r['ok'] else ' '}] {r['texto']}")
    if errores:
        sys.exit(f"\n{errores} etiquetas inválidas: corrígelas antes de entrenar.")
    print("\ndata.yaml generado." + (" Listo para entrenar." if e["listo"] else " Aún no cumple los requisitos (se puede entrenar para pruebas)."))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("iniciar").set_defaults(f=iniciar)
    p = sub.add_parser("importar"); p.add_argument("carpeta"); p.set_defaults(f=importar)
    p = sub.add_parser("dividir"); p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--incluir-preanotadas", action="store_true"); p.set_defaults(f=dividir)
    sub.add_parser("validar").set_defaults(f=validar)
    a = ap.parse_args()
    a.f(a)
