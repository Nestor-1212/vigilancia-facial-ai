"""
Descubrimiento de cámaras IP a partir de IP + usuario + contraseña.

Pasos (cada uno queda registrado para mostrarle al técnico qué pasó):
  1. Puertos: qué servicios responden (HTTP, RTSP…).
  2. ONVIF: marca, modelo y direcciones exactas de video (GetDeviceInformation,
     GetCapabilities, GetProfiles, GetStreamUri) con autenticación WS-Security.
  3. Si no hay ONVIF: rutas RTSP conocidas por marca, comprobadas con RTSP DESCRIBE
     (distingue "ruta no existe" de "credenciales incorrectas" sin abrir el video).
  4. Vista previa: captura un fotograma del stream elegido.

La contraseña nunca se registra en logs ni se devuelve: las URLs se entregan sin credenciales.
"""
import base64
import hashlib
import logging
import os
import re
import secrets
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urlparse, urlunparse

import cv2
import httpx

logger = logging.getLogger(__name__)

PUERTOS = {80: "HTTP", 8080: "HTTP", 8000: "HTTP", 443: "HTTPS", 554: "RTSP", 8554: "RTSP", 37777: "Dahua", 34567: "XMeye"}

# Rutas RTSP conocidas: (marca, principal, secundaria)
RUTAS = [
    ("Hikvision", "/Streaming/Channels/101", "/Streaming/Channels/102"),
    ("Dahua", "/cam/realmonitor?channel=1&subtype=0", "/cam/realmonitor?channel=1&subtype=1"),
    ("Axis", "/axis-media/media.amp", "/axis-media/media.amp?resolution=640x360"),
    ("Uniview", "/unicast/c1/s0/live", "/unicast/c1/s1/live"),
    ("Hanwha", "/profile2/media.smp", "/profile3/media.smp"),
    ("Genérica", "/stream1", "/stream2"),
    ("Genérica", "/live/ch00_0", "/live/ch00_1"),
    ("Genérica", "/h264/ch1/main/av_stream", "/h264/ch1/sub/av_stream"),
    ("Genérica", "/live", None),
    ("Genérica", "/11", "/12"),
    ("Genérica", "/", None),
]


class Registro:
    def __init__(self):
        self.pasos = []

    def add(self, paso: str, ok: bool, detalle: str = ""):
        self.pasos.append({"paso": paso, "ok": ok, "detalle": detalle})


def con_credenciales(url: str, usuario: str, password: str) -> str:
    """Inserta usuario:contraseña (codificados) en una URL rtsp://host/ruta."""
    if not usuario:
        return url
    p = urlparse(url)
    netloc = f"{quote(usuario, safe='')}:{quote(password or '', safe='')}@{p.hostname}" + (f":{p.port}" if p.port else "")
    return urlunparse(p._replace(netloc=netloc))


def sin_credenciales(url: str) -> str:
    p = urlparse(url)
    if not p.hostname:
        return url
    return urlunparse(p._replace(netloc=p.hostname + (f":{p.port}" if p.port else "")))


# ── 1. Puertos ────────────────────────────────────────────────

def _puerto_abierto(ip: str, puerto: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((ip, puerto), timeout=timeout):
            return True
    except OSError:
        return False


def escanear(ip: str) -> list[int]:
    with ThreadPoolExecutor(max_workers=len(PUERTOS)) as ex:
        res = dict(zip(PUERTOS, ex.map(lambda p: _puerto_abierto(ip, p), PUERTOS)))
    return [p for p, ok in res.items() if ok]


# ── 2. ONVIF (SOAP con WS-Security UsernameToken digest) ──────

def _ws_security(usuario: str, password: str) -> str:
    nonce = secrets.token_bytes(16)
    creado = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(hashlib.sha1(nonce + creado.encode() + password.encode()).digest()).decode()
    return (
        '<Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">'
        f'<UsernameToken><Username>{_xml(usuario)}</Username>'
        f'<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password>'
        f'<Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce>'
        f'<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{creado}</Created>'
        "</UsernameToken></Security>"
    )


def _xml(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _soap(url: str, cuerpo: str, usuario: str, password: str, timeout: float = 5) -> tuple[int, str]:
    sobre = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
        'xmlns:tt="http://www.onvif.org/ver10/schema">'
        f"<s:Header>{_ws_security(usuario, password)}</s:Header><s:Body>{cuerpo}</s:Body></s:Envelope>"
    )
    r = httpx.post(url, content=sobre.encode(), timeout=timeout,
                   headers={"Content-Type": "application/soap+xml; charset=utf-8"})
    return r.status_code, r.text


def _tag(xml: str, nombre: str) -> Optional[str]:
    m = re.search(rf"<(?:\w+:)?{nombre}[^>]*>(.*?)</(?:\w+:)?{nombre}>", xml, re.S)
    return m.group(1).strip() if m else None


def _no_autorizado(codigo: int, xml: str) -> bool:
    return codigo in (401, 403) or "NotAuthorized" in xml or "Sender not Authorized" in xml


class CredencialesIncorrectas(Exception):
    pass


def onvif(ip: str, puertos_http: list[int], usuario: str, password: str, reg: Registro) -> Optional[dict]:
    for puerto in puertos_http:
        servicio = f"http://{ip}:{puerto}/onvif/device_service"
        try:
            cod, xml = _soap(servicio, "<tds:GetDeviceInformation/>", usuario, password)
        except httpx.HTTPError:
            continue
        if _no_autorizado(cod, xml):
            reg.add("ONVIF", False, f"La cámara rechazó el usuario o la contraseña (puerto {puerto}).")
            raise CredencialesIncorrectas()
        if cod != 200 or "GetDeviceInformationResponse" not in xml:
            continue
        info = {"fabricante": _tag(xml, "Manufacturer"), "modelo": _tag(xml, "Model"), "firmware": _tag(xml, "FirmwareVersion")}
        reg.add("ONVIF", True, f"{info['fabricante'] or 'Cámara'} {info['modelo'] or ''} (puerto {puerto})".strip())

        # Servicio de medios. Detrás de un router la cámara informa su IP interna:
        # se conserva la ruta y el puerto, pero con la IP que escribió el técnico
        media = servicio
        try:
            cod, xml = _soap(servicio, '<tds:GetCapabilities><tds:Category>Media</tds:Category></tds:GetCapabilities>', usuario, password)
            m = re.search(r"<(?:\w+:)?Media>.*?<(?:\w+:)?XAddr>(.*?)</", xml, re.S)
            if m:
                pm = urlparse(m.group(1).strip())
                media = urlunparse(pm._replace(netloc=f"{ip}:{pm.port or puerto}"))
        except httpx.HTTPError:
            pass

        try:
            cod, xml = _soap(media, "<trt:GetProfiles/>", usuario, password)
        except httpx.HTTPError:
            reg.add("Rutas de video", False, "El servicio de medios ONVIF no respondió")
            return {**info, "metodo": "onvif", "streams": []}
        perfiles = re.findall(r'<(?:\w+:)?Profiles[^>]*token="([^"]+)"[^>]*>(.*?)</(?:\w+:)?Profiles>', xml, re.S)
        streams = []
        for token, cuerpo in perfiles:
            pedido = ('<trt:GetStreamUri><trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream>'
                      '<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup>'
                      f'<trt:ProfileToken>{_xml(token)}</trt:ProfileToken></trt:GetStreamUri>')
            try:
                c2, x2 = _soap(media, pedido, usuario, password)
            except httpx.HTTPError:
                continue
            uri = _tag(x2, "Uri")
            if not uri:
                continue
            ancho, alto = _tag(cuerpo, "Width"), _tag(cuerpo, "Height")
            # Detrás de un router la cámara suele informar su IP interna: se usa la que escribió el técnico
            p = urlparse(sin_credenciales(uri.replace("&amp;", "&")))
            url = urlunparse(p._replace(netloc=ip + (f":{p.port}" if p.port else "")))
            streams.append({
                "nombre": _tag(cuerpo, "Name") or token,
                "url": url,
                "resolucion": f"{ancho}x{alto}" if ancho and alto else None,
                "codec": _tag(cuerpo, "Encoding"),
            })
        if streams:
            reg.add("Rutas de video", True, f"{len(streams)} encontradas por ONVIF")
            return {**info, "metodo": "onvif", "streams": streams}
        reg.add("Rutas de video", False, "ONVIF respondió pero no entregó direcciones de video")
        return {**info, "metodo": "onvif", "streams": []}
    reg.add("ONVIF", False, "La cámara no respondió a ONVIF")
    return None


# ── 3. RTSP DESCRIBE con autenticación Basic/Digest ───────────

def _rtsp_describe(url: str, usuario: str, password: str, timeout: float = 4) -> int:
    """Devuelve el código RTSP (200 existe, 401 credenciales, 404 no existe) o 0 si no conecta."""
    p = urlparse(url)
    host, puerto = p.hostname, p.port or 554

    def pedir(sock, cseq, auth=""):
        msg = f"DESCRIBE {url} RTSP/1.0\r\nCSeq: {cseq}\r\nAccept: application/sdp\r\nUser-Agent: VigiFacial\r\n{auth}\r\n"
        sock.sendall(msg.encode())
        datos = b""
        while b"\r\n\r\n" not in datos:
            trozo = sock.recv(4096)
            if not trozo:
                break
            datos += trozo
        texto = datos.decode(errors="replace")
        m = re.match(r"RTSP/1\.0 (\d+)", texto)
        return (int(m.group(1)) if m else 0), texto

    try:
        with socket.create_connection((host, puerto), timeout=timeout) as s:
            s.settimeout(timeout)
            cod, resp = pedir(s, 1)
            if cod != 401 or not usuario:
                return cod
            cabecera = re.search(r"WWW-Authenticate:\s*(.+)", resp, re.I)
            reto = cabecera.group(1).strip() if cabecera else ""
            if reto.lower().startswith("digest"):
                realm = re.search(r'realm="([^"]*)"', reto).group(1)
                nonce = re.search(r'nonce="([^"]*)"', reto).group(1)
                ha1 = hashlib.md5(f"{usuario}:{realm}:{password}".encode()).hexdigest()
                ha2 = hashlib.md5(f"DESCRIBE:{url}".encode()).hexdigest()
                resp_hash = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
                auth = (f'Authorization: Digest username="{usuario}", realm="{realm}", nonce="{nonce}", '
                        f'uri="{url}", response="{resp_hash}"\r\n')
            else:
                auth = "Authorization: Basic " + base64.b64encode(f"{usuario}:{password}".encode()).decode() + "\r\n"
            cod, _ = pedir(s, 2, auth)
            return cod
    except (OSError, AttributeError):
        return 0


def rutas_rtsp(ip: str, puertos_rtsp: list[int], usuario: str, password: str, reg: Registro, fabricante: Optional[str]) -> Optional[dict]:
    # Si se conoce la marca, sus rutas van primero
    orden = sorted(RUTAS, key=lambda r: 0 if fabricante and r[0].lower() in fabricante.lower() else 1)
    rechazos = 0
    for puerto in puertos_rtsp:
        for marca, principal, secundaria in orden:
            url = f"rtsp://{ip}:{puerto}{principal}"
            cod = _rtsp_describe(url, usuario, password)
            if cod == 401:
                rechazos += 1
                # Muchas cámaras responden 401 a cualquier ruta si la contraseña es incorrecta
                if rechazos >= 3:
                    reg.add("RTSP", False, "La cámara rechazó el usuario o la contraseña.")
                    raise CredencialesIncorrectas()
                continue
            if cod != 200:
                continue
            streams = [{"nombre": "Principal", "url": url, "resolucion": None, "codec": None}]
            if secundaria and _rtsp_describe(f"rtsp://{ip}:{puerto}{secundaria}", usuario, password) == 200:
                streams.append({"nombre": "Secundario", "url": f"rtsp://{ip}:{puerto}{secundaria}", "resolucion": None, "codec": None})
            reg.add("RTSP", True, f"Ruta de video encontrada ({marca if marca != 'Genérica' else 'formato genérico'}, puerto {puerto})")
            return {"fabricante": fabricante or (marca if marca != "Genérica" else None), "modelo": None, "metodo": "rtsp", "streams": streams}
    reg.add("RTSP", False, "Ninguna ruta de video conocida respondió")
    return None


# ── 4. Vista previa ───────────────────────────────────────────

def capturar(url_con_credenciales: str, timeout_s: int = 8) -> Optional[dict]:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;tcp|stimeout;{timeout_s * 1_000_000}"
    cap = cv2.VideoCapture(url_con_credenciales, cv2.CAP_FFMPEG)
    try:
        inicio = time.monotonic()
        while time.monotonic() - inicio < timeout_s:
            ok, frame = cap.read()
            if ok and frame is not None:
                alto, ancho = frame.shape[:2]
                vista = cv2.resize(frame, (640, int(640 * alto / ancho))) if ancho > 640 else frame
                _, jpg = cv2.imencode(".jpg", vista, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return {"resolucion": f"{ancho}x{alto}", "imagen": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}
        return None
    finally:
        cap.release()


# ── Orquestación ──────────────────────────────────────────────

def descubrir(ip: str, usuario: str, password: str) -> dict:
    reg = Registro()
    base = {"ip": ip, "pasos": reg.pasos, "streams": [], "fabricante": None, "modelo": None}

    abiertos = escanear(ip)
    if not abiertos:
        reg.add("Conexión", False, "No respondió ningún puerto")
        return {**base, "ok": False, "error": "no_responde"}
    reg.add("Conexión", True, "Puertos abiertos: " + ", ".join(f"{p} ({PUERTOS[p]})" for p in abiertos))

    http = [p for p in abiertos if PUERTOS[p] == "HTTP"]
    rtsp = [p for p in abiertos if PUERTOS[p] == "RTSP"]
    try:
        resultado = onvif(ip, http, usuario, password, reg) if http else None
        if not resultado or not resultado["streams"]:
            fabricante = resultado.get("fabricante") if resultado else None
            if not rtsp:
                reg.add("RTSP", False, "El puerto de video (554) no está abierto")
                return {**base, **(resultado or {}), "ok": False, "error": "sin_stream", "streams": []}
            resultado = rutas_rtsp(ip, rtsp, usuario, password, reg, fabricante) or resultado
    except CredencialesIncorrectas:
        return {**base, "ok": False, "error": "credenciales"}

    if not resultado or not resultado.get("streams"):
        return {**base, **(resultado or {}), "ok": False, "error": "sin_stream", "streams": []}

    # Vista previa del primer stream (y resolución real de los que no la traen)
    vista = None
    for s in resultado["streams"]:
        cap = capturar(con_credenciales(s["url"], usuario, password))
        if cap:
            s["resolucion"] = s["resolucion"] or cap["resolucion"]
            vista = vista or cap["imagen"]
            s["ok"] = True
        else:
            s["ok"] = False
    reg.add("Imagen", bool(vista), "Se obtuvo una imagen en vivo" if vista else "La ruta existe pero no se pudo leer el video")

    # El stream de menor resolución se recomienda para el análisis (consume menos CPU)
    def pixeles(s):
        try:
            a, b = (s.get("resolucion") or "0x0").lower().split("x")
            return int(a) * int(b)
        except ValueError:
            return 0
    validos = [s for s in resultado["streams"] if s.get("ok")]
    if validos:
        min(validos, key=pixeles)["recomendado"] = True

    return {**base, **resultado, "ok": bool(vista), "error": None if vista else "sin_imagen", "vista_previa": vista}
