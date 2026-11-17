import asyncio
import logging
import socket
import ssl
from uuid import uuid4

import httpcore
import pytest

from dialogentis.integrations.http_client import (
    MAX_BODY,
    IntegrationRequestError,
    _public_address,
    request_json,
)
from dialogentis.integrations.schema import IntegrationInput


def config():
    return IntegrationInput(
        name="Orders",
        base_url="https://orders.example.com/api",
        methods=["GET", "POST"],
        paths=["/orders/", "/status"],
    )


def record(address):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = (address, 443, 0, 0) if family == socket.AF_INET6 else (address, 443)
    return family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr


class RecordingStream(httpcore.AsyncMockStream):
    def __init__(self, response):
        super().__init__([response])
        self.writes = []
        self.tls_host = None
        self.verified_tls = False

    async def write(self, buffer, timeout=None):  # noqa: ASYNC109
        self.writes.append(buffer)

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):  # noqa: ASYNC109
        self.tls_host = server_hostname
        self.verified_tls = (
            ssl_context.check_hostname and ssl_context.verify_mode == ssl.CERT_REQUIRED
        )
        return self


class RecordingBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, response):
        self.stream = RecordingStream(response)
        self.connections = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):  # noqa: ASYNC109
        self.connections.append((host, port))
        return self.stream


def response(status=200, content_type=b"application/json", body=b'{"status":"ok"}', extra=b""):
    return (
        f"HTTP/1.1 {status} Test\r\n".encode()
        + b"Content-Type: "
        + content_type
        + b"\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + extra
        + b"\r\n"
        + body
    )


def use_network(monkeypatch, addresses, raw):
    backend = RecordingBackend(raw)

    async def resolve(host, port, **kwargs):
        assert host == "orders.example.com" and port == 443
        return [record(address) for address in addresses]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
    return backend


async def run_request(method="GET", path="/orders/123", **kwargs):
    return await request_json(
        config(),
        method,
        path,
        kwargs.pop("query", {}),
        kwargs.pop("body", None),
        secret=kwargs.pop("secret", None),
        execution_id=kwargs.pop("execution_id", uuid4()),
        supports_idempotency=kwargs.pop("supports_idempotency", False),
        remaining_seconds=kwargs.pop("remaining_seconds", 5),
        **kwargs,
    )


async def test_pins_dns_and_preserves_tls_identity(monkeypatch):
    backend = use_network(monkeypatch, ["8.8.8.8", "2606:4700:4700::1111"], response())
    execution_id = uuid4()
    result = await run_request(
        "POST",
        query={"q": "a&b"},
        body={"order": 1},
        secret="token-value",
        execution_id=execution_id,
        supports_idempotency=True,
    )
    assert result.value == {"status": "ok"} and "ok" not in repr(result)
    assert backend.connections == [("8.8.8.8", 443)]
    assert backend.stream.tls_host == "orders.example.com" and backend.stream.verified_tls
    request = b"".join(backend.stream.writes)
    assert b"POST /api/orders/123?q=a%26b HTTP/1.1" in request
    assert b"host: orders.example.com" in request.lower()
    assert b"authorization: bearer token-value" in request.lower()
    assert str(execution_id).encode() in request


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "::1",
        "::ffff:8.8.8.8",
        "64:ff9b::808:808",
        "2002:0808:0808::1",
        "ff02::1",
    ],
)
def test_blocks_nonpublic_and_transition_addresses(address):
    with pytest.raises(IntegrationRequestError) as error:
        _public_address(address)
    assert error.value.code == "address_blocked"


async def test_rejects_mixed_dns_without_connecting(monkeypatch):
    backend = use_network(monkeypatch, ["8.8.8.8", "127.0.0.1"], response())
    with pytest.raises(IntegrationRequestError) as error:
        await run_request()
    assert error.value.code == "address_blocked" and not backend.connections


async def test_rejects_excessive_dns_answers(monkeypatch):
    backend = use_network(monkeypatch, ["8.8.8.8"] * 101, response())
    with pytest.raises(IntegrationRequestError) as error:
        await run_request()
    assert error.value.code == "address_blocked" and not backend.connections


async def test_debug_logs_do_not_expose_request_or_response_data(monkeypatch, caplog):
    use_network(
        monkeypatch,
        ["8.8.8.8"],
        response(body=b'{"value":"private-response"}', extra=b"Set-Cookie: private-cookie\r\n"),
    )
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("httpcore.http11").debug("outside integration request")
        result = await run_request(
            "POST",
            query={"q": "private-query"},
            body={"value": "private-body"},
            secret="private-token",
        )
    assert result.value == {"value": "private-response"}
    assert "outside integration request" in caplog.text
    for value in (
        "private-response",
        "private-cookie",
        "private-query",
        "private-body",
        "private-token",
    ):
        assert value not in caplog.text


@pytest.mark.parametrize(
    "path", ["/orders2/1", "/orders/../private", "/orders/%2fprivate", "//other.example/path"]
)
async def test_rejects_paths_outside_allowlist(path):
    with pytest.raises(IntegrationRequestError) as error:
        await run_request(path=path)
    assert error.value.code in {"request_not_allowed", "invalid_request"}


@pytest.mark.parametrize(
    "raw, expected",
    [
        (response(302, extra=b"Location: https://evil.example/\r\n"), "http_error"),
        (response(content_type=b"text/html"), "invalid_response"),
        (response(content_type=b"text/fake+json"), "invalid_response"),
        (response(body=b"not json"), "invalid_response"),
        (response(body=b'{"n":NaN}'), "invalid_response"),
        (response(body=b"x" * (MAX_BODY + 1)), "invalid_response"),
        (response(extra=b"Content-Encoding: gzip\r\n"), "invalid_response"),
    ],
)
async def test_rejects_redirects_and_invalid_responses(monkeypatch, raw, expected):
    backend = use_network(monkeypatch, ["8.8.8.8"], raw)
    with pytest.raises(IntegrationRequestError) as error:
        await run_request()
    assert error.value.code == expected and len(backend.connections) == 1


async def test_mutating_timeout_has_unknown_outcome_and_is_not_retried(monkeypatch):
    backend = use_network(monkeypatch, ["8.8.8.8"], response())

    async def stuck(host, port, **kwargs):
        backend.connections.append((host, port))
        await asyncio.Event().wait()

    monkeypatch.setattr(backend, "connect_tcp", stuck)
    with pytest.raises(IntegrationRequestError) as error:
        await run_request("POST", remaining_seconds=0.05)
    assert error.value.code == "unknown_result" and len(backend.connections) == 1


async def test_get_network_timeout_reports_timeout(monkeypatch):
    backend = use_network(monkeypatch, ["8.8.8.8"], response())

    async def timeout(host, port, **kwargs):
        raise httpcore.ConnectTimeout("private detail")

    monkeypatch.setattr(backend, "connect_tcp", timeout)
    with pytest.raises(IntegrationRequestError) as error:
        await run_request()
    assert error.value.code == "timeout" and "private detail" not in str(error.value)
