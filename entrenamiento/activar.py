"""
Activa un modelo candidato (o vuelve al anterior) y avisa al servicio de IA para recargarlo.

  venv\\python.exe entrenamiento\\activar.py <id> [--forzar]
  venv\\python.exe entrenamiento\\activar.py --anterior
  venv\\python.exe entrenamiento\\activar.py --listar

Sin --forzar, solo se activan modelos que cumplen los criterios de producción.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.registro_modelos import activar, listar_candidatos, volver_al_anterior  # noqa: E402


def avisar_servicio():
    try:
        import httpx
        r = httpx.post("http://localhost:8001/modelos/recargar", timeout=120)
        print("Servicio de IA recargado." if r.status_code == 200 else f"El servicio respondió {r.status_code}: reinícialo.")
    except Exception:
        print("El servicio de IA no responde: se usará el nuevo modelo al reiniciarlo.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", nargs="?")
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--anterior", action="store_true")
    ap.add_argument("--listar", action="store_true")
    a = ap.parse_args()

    if a.listar or not (a.id or a.anterior):
        for c in listar_candidatos():
            estado = "ACTIVO" if c["activo"] else ("apto" if c["apto_produccion"] else "no apto")
            m = f"{c['map50']:.2f}" if c["map50"] is not None else "—"
            print(f"  {c['id']:<24} mAP@0.5={m:<6} {estado:<8} {c['nombre']}")
        return
    try:
        r = volver_al_anterior() if a.anterior else activar(a.id, a.forzar)
    except (FileNotFoundError, ValueError, PermissionError) as e:
        sys.exit(str(e) + ("\nUsa --forzar para activarlo igualmente (solo para pruebas)." if isinstance(e, PermissionError) else ""))
    print(f"Modelo activo: {r['id']}")
    avisar_servicio()


if __name__ == "__main__":
    main()
