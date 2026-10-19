# Entrenar un modelo de EPP para un cliente

Proceso para pasar del modelo de referencia (no apto para producción) a un modelo entrenado
con fotos reales del cliente. Todos los comandos se ejecutan desde `vigilancia-facial-ai/`.

> **Privacidad.** Las fotos muestran trabajadores identificables. Antes de recolectarlas, confirma
> que el cliente tiene el consentimiento o la base legal para usarlas en entrenamiento, y dónde
> se pueden almacenar.

## 1. Preparar la estructura

```powershell
venv\python.exe entrenamiento\preparar.py iniciar
```

Crea `entrenamiento/dataset/` y `clases.json` con las clases del modelo activo. Edita `clases.json`
para dejar solo las que el cliente necesita y agregar las nuevas. Ejemplo para el uniforme:

```json
{
  "helmet":            { "elemento": "casco",    "presente": true },
  "no_helmet":         { "elemento": "casco",    "presente": false },
  "shoes":             { "elemento": "botas",    "presente": true },
  "uniforme_cliente":  { "elemento": "uniforme", "presente": true }
}
```

- La clave es el nombre de la clase en el modelo; `elemento` debe existir en el catálogo
  (Configuración → Elementos detectables → **Nuevo elemento** para crear "uniforme").
- `presente: false` marca clases de ausencia ("sin casco"). Son opcionales: el sistema también
  infiere la falta cuando no ve el elemento en su parte del cuerpo.

## 2. Reunir fotos

Hay dos fuentes, y conviene combinarlas:

- **Importar fotos o fotogramas** del cliente:
  `venv\python.exe entrenamiento\preparar.py importar C:\ruta\a\fotos`
- **Muestras desde las alertas**: en Configuración → Sistema, activa "Recolectar muestras de
  entrenamiento". Cada alerta validada por un operador (sobre todo los **falsos positivos**) se
  copia a `dataset/pendientes/` con su contexto.

Objetivo mínimo: **200 ejemplos por elemento**, con variedad de cámaras, horas, iluminación,
distancias y colores de equipo, y fotos de personas **con y sin** cada elemento.

## 3. Etiquetar

1. Preanota con el modelo activo para no dibujar desde cero:
   `venv\python.exe entrenamiento\preanotar.py`
2. Abre `dataset/pendientes/` en una herramienta que acepte formato YOLO
   (Label Studio, CVAT o Roboflow), corrige cada recuadro y exporta en formato YOLO a la misma carpeta.
3. En el `.json` de cada foto revisada, cambia `"revisada": false` por `"revisada": true`.

## 4. Repartir y validar

```powershell
venv\python.exe entrenamiento\preparar.py dividir --val 0.2
venv\python.exe entrenamiento\preparar.py validar
```

`validar` revisa las etiquetas, genera `data.yaml` y muestra qué requisitos faltan.

## 5. Entrenar

```powershell
venv\python.exe entrenamiento\entrenar.py --id cliente_x_v1 --nombre "Cliente X v1" --epocas 100 --dispositivo 0
```

- `--dispositivo 0` usa la GPU (muy recomendado). En CPU funciona, pero puede tardar horas.
- Por defecto afina el modelo activo (`--base activo`). Para empezar de un modelo general: `--base yolov8s.pt`.
- El resultado queda como **candidato** en `modelos_epp/candidatos/cliente_x_v1/`, con su precisión
  por clase. No se activa solo.

Para comparar con las mismas fotos, mide también los demás candidatos (incluido el de referencia):

```powershell
venv\python.exe entrenamiento\evaluar.py
```

## 6. Activar

Desde **Configuración → Modelos y entrenamiento**, o:

```powershell
venv\python.exe entrenamiento\activar.py cliente_x_v1
venv\python.exe entrenamiento\activar.py --anterior     # volver atrás
```

Solo se activan modelos que cumplen los criterios de producción (mAP@0.5 ≥ 0.70, AP ≥ 0.60 en cada
elemento, al menos 50 imágenes de validación). `--forzar` permite activarlo igualmente para pruebas.

## 7. Medir en operación

Con el modelo activo, los operadores validan las alertas con "Correcta" o "Falso positivo".
Esa precisión real aparece en Configuración → Elementos detectables, y un elemento pasa a
**Validado** al superar el 90 % en 50 validaciones. Los falsos positivos recolectados alimentan
la siguiente versión del modelo: vuelve al paso 3.
