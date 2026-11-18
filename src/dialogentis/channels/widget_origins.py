import ipaddress
import re
from urllib.parse import urlsplit

import idna

LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def widget_origin(value: str, *, development: bool) -> str:
    value = value.strip(" ")
    if not value or any(char.isspace() or not char.isprintable() for char in value):
        raise ValueError("Invalid origin")
    if any(char in value for char in ("*", "\\", "@", "#", "?", "%")):
        raise ValueError("Invalid origin")
    parsed = urlsplit(value)
    host, port = parsed.hostname, parsed.port
    if (
        not host
        or parsed.scheme not in {"https", "http"}
        or parsed.path not in {"", "/"}
        or port == 0
        or parsed.netloc.endswith(":")
        or parsed.netloc.startswith("[")
        and re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", parsed.netloc) is None
    ):
        raise ValueError("Invalid origin")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        host = idna.encode(host, uts46=True, std3_rules=True).decode("ascii").lower()
        labels = host.removesuffix(".").split(".")
        if (
            len(host) > 253
            or not all(LABEL.fullmatch(label) for label in labels)
            or not labels[-1][0].isalpha()
        ):
            raise ValueError("Invalid origin") from None
        local = host.removesuffix(".").split(".")[-1] == "localhost"
    else:
        local = address.is_loopback or (
            isinstance(address, ipaddress.IPv6Address)
            and address.ipv4_mapped is not None
            and address.ipv4_mapped.is_loopback
        )
        host = f"[{address}]" if address.version == 6 else str(address)
    if local and not development or parsed.scheme == "http" and not local:
        raise ValueError("Invalid origin")
    default_port = 443 if parsed.scheme == "https" else 80
    suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{parsed.scheme}://{host}{suffix}"
