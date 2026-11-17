import asyncio
import os
import socket
import ssl
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpcore._async.connection as connection_module
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from httpcore._backends.anyio import AnyIOBackend

from dialogentis.integrations.http_client import IntegrationRequestError, request_json
from dialogentis.integrations.schema import IntegrationInput

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="Run with Compose test profile"
    ),
]


async def test_real_tls_verifies_original_hostname(monkeypatch, tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "orders.example.com")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("orders.example.com")]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    names = []
    server_context.set_servername_callback(
        lambda _socket, hostname, _context: names.append(hostname)
    )

    async def serve(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b'Content-Length: 11\r\nConnection: close\r\n\r\n{"ok":true}'
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0, ssl=server_context)
    local_port = server.sockets[0].getsockname()[1]
    loop = asyncio.get_running_loop()
    real_resolver = loop.getaddrinfo
    real_connect = AnyIOBackend.connect_tcp

    async def resolver(host, port, **kwargs):
        if host in {"orders.example.com", "wrong.example.com"}:
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", port))]
        return await real_resolver(host, port, **kwargs)

    async def connect(self, host, port, **kwargs):
        assert host == "8.8.8.8" and port == 443
        return await real_connect(self, "127.0.0.1", local_port, **kwargs)

    monkeypatch.setattr(loop, "getaddrinfo", resolver)
    monkeypatch.setattr(AnyIOBackend, "connect_tcp", connect)
    monkeypatch.setattr(
        connection_module,
        "default_ssl_context",
        lambda: ssl.create_default_context(cafile=str(cert_path)),
    )
    try:

        def settings(host):
            return IntegrationInput(
                name="Orders", base_url=f"https://{host}", methods=["GET"], paths=["/status"]
            )

        async def call(host):
            return await request_json(
                settings(host),
                "GET",
                "/status",
                {},
                None,
                secret=None,
                execution_id=uuid4(),
                supports_idempotency=False,
                remaining_seconds=5,
            )

        assert (await call("orders.example.com")).value == {"ok": True}
        with pytest.raises(IntegrationRequestError) as error:
            await call("wrong.example.com")
        assert error.value.code == "network_error"
        assert names == ["orders.example.com", "wrong.example.com"]
    finally:
        server.close()
        await server.wait_closed()
