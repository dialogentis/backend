"""Bounded HTTPS execution for trusted integration configuration.

The caller must persist an execution ledger before any state-changing request.
This module is not connected to runtime until that ledger exists.
"""

import asyncio
import contextvars
import ipaddress
import json
import logging
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import quote, unquote, urlencode, urlsplit
from uuid import UUID

import httpcore

from dialogentis.integrations.schema import IntegrationInput, safe_base_url, safe_path

MAX_BODY = 256 * 1024
MAX_QUERY = 8192
_TRANSITION_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32")
)
_sensitive_request = contextvars.ContextVar("integration_sensitive_request", default=False)


class _SensitiveTraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (_sensitive_request.get() and record.levelno <= logging.DEBUG)


_trace_filter = _SensitiveTraceFilter()
for _logger_name in ("httpcore.connection", "httpcore.http11"):
    logging.getLogger(_logger_name).addFilter(_trace_filter)


@asynccontextmanager
async def _suppress_sensitive_trace():
    token = _sensitive_request.set(True)
    try:
        yield
    finally:
        _sensitive_request.reset(token)


class IntegrationRequestError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class IntegrationResponse:
    status: int
    value: Any = field(repr=False)


def _public_address(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise IntegrationRequestError("address_blocked") from None
    if (
        not address.is_global
        or address.is_multicast
        or address.is_unspecified
        or (
            isinstance(address, ipaddress.IPv6Address)
            and (
                address.scope_id is not None
                or address.ipv4_mapped is not None
                or any(address in network for network in _TRANSITION_NETWORKS)
            )
        )
    ):
        raise IntegrationRequestError("address_blocked")
    return str(address)


async def _resolve(host: str, port: int) -> tuple[str, ...]:
    try:
        records = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except OSError:
        raise IntegrationRequestError("dns_error") from None
    if len(records) > 100:
        raise IntegrationRequestError("address_blocked")
    addresses: list[str] = []
    for family, socktype, protocol, _, sockaddr in records:
        if (
            family not in (socket.AF_INET, socket.AF_INET6)
            or socktype != socket.SOCK_STREAM
            or protocol != socket.IPPROTO_TCP
        ):
            raise IntegrationRequestError("address_blocked")
        if not isinstance(sockaddr[0], str):
            raise IntegrationRequestError("address_blocked")
        if family == socket.AF_INET6 and (len(sockaddr) != 4 or sockaddr[3] != 0):
            raise IntegrationRequestError("address_blocked")
        address = _public_address(sockaddr[0])
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise IntegrationRequestError("dns_error")
    return tuple(addresses)


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, host: str, port: int, addresses: tuple[str, ...]):
        self.host, self.port, self.addresses = host, port, addresses
        self.backend = cast(httpcore.AsyncNetworkBackend, httpcore.AnyIOBackend())

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):  # noqa: ASYNC109
        if host != self.host or port != self.port or local_address is not None:
            raise IntegrationRequestError("address_blocked")
        last_error = None
        for address in self.addresses:
            try:
                return await self.backend.connect_tcp(
                    address, port, timeout=timeout, socket_options=socket_options
                )
            except httpcore.ConnectError as exc:
                last_error = exc
        raise httpcore.ConnectError("Unable to connect to a verified address") from last_error

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


def _request_url(config: IntegrationInput, method: str, path: str, query: dict[str, str]) -> str:
    if (
        not config.enabled
        or not isinstance(method, str)
        or method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}
        or method not in config.methods
    ):
        raise IntegrationRequestError("request_not_allowed")
    try:
        base = safe_base_url(config.base_url)
        decoded = unquote(safe_path(path), errors="strict")
        allowed = [unquote(safe_path(item), errors="strict") for item in config.paths]
        if not any(
            decoded == item or (item.endswith("/") and decoded.startswith(item)) for item in allowed
        ):
            raise IntegrationRequestError("request_not_allowed")
        if (
            not isinstance(query, dict)
            or len(query) > 100
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or len(key) > 256
                or len(value) > 2048
                for key, value in query.items()
            )
        ):
            raise IntegrationRequestError("invalid_request")
        encoded_query = urlencode(query)
        if len(encoded_query.encode("ascii")) > MAX_QUERY:
            raise IntegrationRequestError("invalid_request")
        return base + quote(decoded, safe="/-._~") + (f"?{encoded_query}" if encoded_query else "")
    except ValueError, UnicodeError, AttributeError, TypeError:
        raise IntegrationRequestError("invalid_request") from None


async def request_json(
    config: IntegrationInput,
    method: str,
    path: str,
    query: dict[str, str],
    body: Any,
    *,
    secret: str | None,
    execution_id: UUID,
    supports_idempotency: bool,
    remaining_seconds: float,
) -> IntegrationResponse:
    if (
        not isinstance(execution_id, UUID)
        or type(supports_idempotency) is not bool
        or type(remaining_seconds) not in (int, float)
        or not 0 < remaining_seconds <= 20
    ):
        raise IntegrationRequestError("invalid_request")
    url = _request_url(config, method, path, query)
    try:
        content = (
            None
            if body is None
            else json.dumps(
                body, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        )
    except TypeError, ValueError, UnicodeError, RecursionError:
        raise IntegrationRequestError("invalid_request") from None
    if content is not None and len(content) > MAX_BODY:
        raise IntegrationRequestError("invalid_request")
    headers = [(b"Accept", b"application/json"), (b"Accept-Encoding", b"identity")]
    if content is not None:
        headers.append((b"Content-Type", b"application/json"))
    if secret is not None:
        try:
            encoded = secret.encode("ascii")
        except AttributeError, UnicodeError:
            raise IntegrationRequestError("invalid_credential") from None
        if not encoded or len(encoded) > 2048 or any(char < 32 or char >= 127 for char in encoded):
            raise IntegrationRequestError("invalid_credential")
        headers.append((b"Authorization", b"Bearer " + encoded))
    if method != "GET" and supports_idempotency:
        headers.append((b"Idempotency-Key", str(execution_id).encode("ascii")))
    parsed = urlsplit(url)
    assert parsed.hostname is not None
    host, port = parsed.hostname, parsed.port or 443
    sending = False
    try:
        async with _suppress_sensitive_trace(), asyncio.timeout(min(8, remaining_seconds)):
            addresses = await _resolve(host, port)
            backend = _PinnedBackend(host, port, addresses)
            async with httpcore.AsyncConnectionPool(
                network_backend=backend, max_connections=1, max_keepalive_connections=0, retries=0
            ) as pool:
                sending = True
                async with pool.stream(method, url, headers=headers, content=content) as response:
                    if not 200 <= response.status < 300:
                        raise IntegrationRequestError("http_error")
                    content_types = [
                        value for name, value in response.headers if name.lower() == b"content-type"
                    ]
                    encodings = [
                        value
                        for name, value in response.headers
                        if name.lower() == b"content-encoding"
                    ]
                    media_type = (
                        content_types[0].split(b";", 1)[0].strip().lower()
                        if len(content_types) == 1
                        else b""
                    )
                    if (
                        len(content_types) != 1
                        or not (
                            media_type == b"application/json"
                            or (
                                media_type.startswith(b"application/")
                                and media_type.endswith(b"+json")
                            )
                        )
                        or any(value.strip().lower() != b"identity" for value in encodings)
                    ):
                        raise IntegrationRequestError("invalid_response")
                    data = bytearray()
                    async for chunk in response.aiter_stream():
                        data.extend(chunk)
                        if len(data) > MAX_BODY:
                            raise IntegrationRequestError("invalid_response")
                    try:
                        value = json.loads(
                            data.decode("utf-8"),
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
                        )
                    except UnicodeError, ValueError, RecursionError:
                        raise IntegrationRequestError("invalid_response") from None
                    return IntegrationResponse(response.status, value)
    except TimeoutError:
        raise IntegrationRequestError(
            "unknown_result" if sending and method != "GET" else "timeout"
        ) from None
    except httpcore.TimeoutException:
        raise IntegrationRequestError(
            "unknown_result" if sending and method != "GET" else "timeout"
        ) from None
    except httpcore.NetworkError, httpcore.ProtocolError:
        raise IntegrationRequestError(
            "unknown_result" if sending and method != "GET" else "network_error"
        ) from None
