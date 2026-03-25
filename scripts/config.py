import os


def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """
    Read environment variable with optional fail-fast behavior.
    """
    value = os.environ.get(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

