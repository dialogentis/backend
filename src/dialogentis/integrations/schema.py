"""Restrictive integration configuration; network resolution is checked at execution."""

import ipaddress
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import Field, SecretStr, StrictBool, StrictStr, field_validator

from dialogentis.contracts import InputModel

Method = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


def safe_path(value: str) -> str:
    if not value.startswith("/") or value.startswith("//") or len(value) > 1024:
        raise ValueError("Invalid path")
    decoded = unquote(value, errors="strict")
    if (
        any(char in decoded for char in "\\?#%")
        or decoded.count("/") != value.count("/")
        or any(ord(char) < 32 or ord(char) == 127 for char in decoded)
        or any(segment in (".", "..") for segment in decoded.split("/"))
        or "//" in decoded
    ):
        raise ValueError("Invalid path")
    return value


def path_allowed(path: str, allowed_paths: list[str]) -> bool:
    decoded = unquote(safe_path(path), errors="strict")
    allowed = [unquote(safe_path(item), errors="strict") for item in allowed_paths]
    return any(
        decoded == item or (item.endswith("/") and decoded.startswith(item)) for item in allowed
    )


def safe_base_url(value: str) -> str:
    if len(value) > 2048 or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        raise ValueError("Invalid URL")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if (
            parsed.scheme != "https"
            or not host
            or "%" in host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "\\" in parsed.netloc
            or port == 0
        ):
            raise ValueError("Invalid URL")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            ascii_host = host.encode("idna").decode("ascii").lower()
            if (
                len(ascii_host) > 253
                or ascii_host.endswith(".")
                or any(not label or len(label) > 63 for label in ascii_host.split("."))
                or any(
                    not (char.isascii() and (char.isalnum() or char in ".-")) for char in ascii_host
                )
                or any(
                    label.startswith("-") or label.endswith("-") for label in ascii_host.split(".")
                )
            ):
                raise ValueError("Invalid hostname") from None
            host = ascii_host
        else:
            if not address.is_global or address.is_multicast:
                raise ValueError("Non-public address")
            host = f"[{address}]" if address.version == 6 else str(address)
        if parsed.path:
            safe_path(parsed.path)
        return f"https://{host}{f':{port}' if port is not None else ''}{parsed.path.rstrip('/')}"
    except ValueError, UnicodeError:
        raise ValueError("Invalid URL") from None


class IntegrationInput(InputModel):
    name: StrictStr = Field(min_length=1, max_length=120)
    base_url: StrictStr = Field(min_length=1, max_length=2048)
    methods: list[Method] = Field(min_length=1, max_length=5)
    paths: list[StrictStr] = Field(min_length=1, max_length=100)
    response_fields: list[StrictStr] = Field(default_factory=list, max_length=100)
    secret: SecretStr | None = Field(default=None, repr=False)
    enabled: StrictBool = True

    @field_validator("name")
    @classmethod
    def name_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Invalid name")
        return value

    @field_validator("base_url")
    @classmethod
    def valid_base(cls, value: str) -> str:
        return safe_base_url(value)

    @field_validator("paths")
    @classmethod
    def valid_paths(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Duplicate path")
        return [safe_path(path) for path in value]

    @field_validator("methods", "response_fields")
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value) or any(not item or len(item) > 128 for item in value):
            raise ValueError("Invalid values")
        return value
