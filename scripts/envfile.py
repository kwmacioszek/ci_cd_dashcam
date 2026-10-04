"""Minimalny loader .env (bez python-dotenv): ustawia zmienne, które nie są jeszcze w środowisku."""

import os
from pathlib import Path


def load_env(path: Path | str = ".env") -> list[str]:
    """Wczytuje KLUCZ=wartość z pliku; zwraca nazwy ustawionych kluczy (nigdy wartości)."""
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if val and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded
