# Modelos de EPP

El servicio carga el modelo definido en `config.json` (campo `archivo`).
Los pesos `.pt` no se guardan en git.

## Modelo de referencia (solo desarrollo)

```powershell
Invoke-WebRequest "https://huggingface.co/keremberke/yolov8s-protective-equipment-detection/resolve/main/best.pt" -OutFile referencia_yolov8s_epp.pt
```

- Clases: glove, goggles, helmet, mask, shoes y sus ausencias (`no_*`). No detecta chaleco.
- mAP@0.5 = 0.28 y sin licencia declarada: **no apto para producción**.

## Cambiar de modelo

1. Copia el `.pt` a esta carpeta.
2. En `config.json` ajusta `archivo`, `nombre`, `apto_produccion`, `advertencia` y el mapa `clases`
   (nombre de clase del modelo → elemento del catálogo, y si indica presencia o ausencia).
3. Reinicia el servicio de IA. El catálogo de Laravel se actualiza solo al consultar las capacidades del servicio.
