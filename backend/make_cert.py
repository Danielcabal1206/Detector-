"""
make_cert.py — Genera un certificado HTTPS autofirmado para la red local.

Por qué hace falta: Safari en iPhone sólo permite getUserMedia (la cámara) en
"contextos seguros": HTTPS, o http://localhost. Como el iPhone accede por la IP
de la PC (http://192.168.x.x), necesitas HTTPS.

Uso (desde backend/):
    python make_cert.py                 # detecta las IPs de esta PC automáticamente
    python make_cert.py 192.168.1.50    # o añade IPs / nombres extra al certificado

Crea certs/cert.pem y certs/key.pem; main.py los usa solo si existen.
"""
import datetime
import ipaddress
import socket
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

CERT_DIR = Path(__file__).resolve().parent / "certs"


def lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def main(extra: list[str]) -> None:
    names: list[x509.GeneralName] = [x509.DNSName("localhost"),
                                     x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    shown = ["localhost", "127.0.0.1"]
    for item in lan_ips() + extra:
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(item)))
        except ValueError:
            names.append(x509.DNSName(item))
        shown.append(item)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "restaurant-monitor.local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )

    CERT_DIR.mkdir(exist_ok=True)
    (CERT_DIR / "key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    (CERT_DIR / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Certificado creado en {CERT_DIR}")
    print("Válido para: " + ", ".join(shown))
    print("Si cambias de red (otra IP), vuelve a ejecutar este script.")


if __name__ == "__main__":
    main(sys.argv[1:])
