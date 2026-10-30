"""Resolve the OpenRouter credential without copying it into files or logs."""
import os
import subprocess
from .config import CredentialSettings


def openrouter_api_key(settings: CredentialSettings | None = None) -> str:
    """Prefer the user's Bash startup; fall back to the process environment.

    Interactive Bash handles .bashrc's common non-interactive early-return guard.
    Startup output is captured and discarded; NUL delimiters isolate the key.
    The user's own startup configuration is trusted executable shell code.
    """
    settings = settings or CredentialSettings()
    variable = settings.env_var
    environment_key = os.environ.get(variable, "").strip()
    if environment_key and not settings.prefer_bashrc:
        return environment_key
    if settings.bashrc.expanduser().is_file():
        try:
            shell_env = os.environ.copy()
            shell_env.pop(variable, None)
            result = subprocess.run(
                [settings.bash_executable, "--noprofile", "--rcfile", str(settings.bashrc.expanduser()), "-ic",
                 'printf "\\0%s\\0" "${' + variable + '-}"'],
                env=shell_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=settings.timeout_seconds,
                check=False,
            )
            parts = result.stdout.rsplit(b"\0", 2)
            if result.returncode == 0 and len(parts) == 3:
                key = parts[1].decode("utf-8").strip()
                if key:
                    return key
        except (OSError, subprocess.TimeoutExpired, UnicodeError):
            pass
    if environment_key:
        return environment_key
    raise ValueError(
        f"{variable} is missing. Export it in {settings.bashrc} or your current "
        "shell. Retrieval-only commands need no credentials."
    )
