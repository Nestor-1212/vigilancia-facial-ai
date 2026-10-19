"""
Genera un video SINTÉTICO de demostración para probar la vista "En vivo"
sin cámaras reales. No contiene personas reales ni rostros.

Uso:  venv\\python.exe demo_videos\\generar_demo.py
Luego, en Cámaras, usa como URL/Fuente la ruta absoluta del .mp4 generado.
"""
from pathlib import Path

import cv2
import numpy as np

W, H, FRAMES = 1280, 720, 120   # el worker lee 1 frame/s → 2 minutos en bucle
OUT = Path(__file__).with_name("estacionamiento_demo.mp4")


def fondo() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    # Piso con degradado (gris azulado, más claro al frente)
    for y in range(H):
        t = y / H
        img[y, :] = (int(48 + 40 * t), int(52 + 38 * t), int(56 + 34 * t))
    # Pared / fondo
    img[:220, :] = (38, 40, 44)
    cv2.rectangle(img, (0, 205), (W, 222), (70, 74, 80), -1)
    # Luminarias
    for x in range(140, W, 250):
        cv2.rectangle(img, (x - 50, 30), (x + 50, 42), (210, 225, 230), -1)
        cv2.circle(img, (x, 120), 140, (60, 64, 70), -1, lineType=cv2.LINE_AA)
    img[:220, :] = cv2.GaussianBlur(img[:220, :], (0, 0), 25)
    # Líneas de estacionamiento en perspectiva
    for i in range(-6, 8):
        x_top = 640 + i * 110
        x_bot = 640 + i * 260
        cv2.line(img, (x_top, 230), (x_bot, H), (170, 175, 160), 3, cv2.LINE_AA)
    cv2.line(img, (0, 470), (W, 470), (150, 155, 140), 2, cv2.LINE_AA)
    # Columna
    cv2.rectangle(img, (930, 150), (990, 560), (88, 92, 98), -1)
    cv2.rectangle(img, (930, 300), (990, 330), (40, 170, 210), -1)
    return img


def vehiculo(img, x, y, color):
    cv2.rectangle(img, (x, y), (x + 260, y + 90), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x + 50, y - 50), (x + 200, y + 5), tuple(int(c * 0.8) for c in color), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x + 62, y - 40), (x + 188, y), (90, 70, 50), -1)
    for cx in (x + 55, x + 205):
        cv2.circle(img, (cx, y + 92), 26, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x + 245, y + 25), (x + 260, y + 45), (120, 220, 255), -1)


def peaton(img, x, y, escala, color):
    h = int(150 * escala)
    cv2.ellipse(img, (x, y + h), (int(30 * escala), int(8 * escala)), 0, 0, 360, (35, 38, 42), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x - int(18 * escala), y + int(40 * escala)), (x + int(18 * escala), y + int(105 * escala)), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x - int(15 * escala), y + int(105 * escala)), (x + int(15 * escala), y + h), (50, 50, 60), -1)
    cv2.circle(img, (x, y + int(22 * escala)), int(17 * escala), (95, 120, 150), -1, cv2.LINE_AA)


def main():
    base = fondo()
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 1, (W, H))
    for f in range(FRAMES):
        img = base.copy()
        t = f / FRAMES
        vehiculo(img, int(-300 + t * 1900), 330, (150, 110, 40))
        peaton(img, int(180 + 6 * f), 300, 1.0, (60, 90, 170))
        peaton(img, int(1150 - 4 * (f % 60) * 2), 340, 1.15, (70, 140, 80))
        # Ruido leve de sensor
        ruido = np.random.default_rng(f).normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
        cv2.putText(img, "DEMO - VIDEO SINTETICO", (W - 330, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    print(f"Video generado: {OUT} ({FRAMES} frames)")


if __name__ == "__main__":
    main()
