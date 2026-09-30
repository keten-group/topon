"""Which process writes a run directory, and whether it is still alive.

Long sweeps lost time to two processes writing one directory: each
re-ran the other's cases and their files mixed. A run records who is
writing it (pid, host, start time) where the next writer can read it,
and a writer that finds another live one can say so.

``pid_alive`` is here rather than a psutil call so topon keeps no extra
dependency. On Windows it asks the kernel directly: ``os.kill(pid, 0)``
there is not a probe but sends CTRL_C_EVENT, which stops the process.

The host is recorded as a short hash of its name (:func:`host_id`), not the
name itself: the check only needs "same machine or not", and a manifest
committed with a demo's expected output or shared with a dataset would
otherwise carry the machine's name. Records written before this, with the
plain name, are still recognised.
"""
from __future__ import annotations

import hashlib
import os
import socket
from datetime import datetime, timezone
from typing import Optional


def pid_alive(pid) -> bool:
    """True when a process with this id is running on this machine.

    A pid can be reused after its process exits, so a True here is
    "something with that id is running", which is why callers compare the
    host and warn rather than refuse.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        query_limited_information = 0x1000
        still_active = 259
        access_denied = 5
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(query_limited_information, False, pid)
        if not handle:
            # A process we may not query still exists.
            return ctypes.get_last_error() == access_denied
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def host_id() -> str:
    """This machine as ``"h-"`` and 12 hex digits of a hash of its name."""
    name = socket.gethostname().encode("utf-8", "replace")
    return "h-" + hashlib.sha256(name).hexdigest()[:12]


def _this_host(host) -> bool:
    """True when a recorded ``host`` is this machine, hashed or (older) plain."""
    return host in (host_id(), socket.gethostname())


def this_process() -> dict:
    """``{"pid", "host", "started"}`` for the calling process, now."""
    return {
        "pid": os.getpid(),
        "host": host_id(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def other_live_writer(record: Optional[dict]) -> Optional[dict]:
    """``record`` when it names another process on this host that is alive.

    ``record`` is what :func:`this_process` wrote for an earlier writer.
    A record from another host cannot be checked from here and is not
    reported; one naming this process, or a process that has exited, is
    a finished writer, not a competing one.
    """
    if not isinstance(record, dict):
        return None
    if not _this_host(record.get("host")):
        return None
    if record.get("pid") == os.getpid():
        return None
    return record if pid_alive(record.get("pid")) else None


def stop_hint(pid) -> str:
    """How to look at and stop process ``pid`` from a shell on this OS."""
    if os.name == "nt":
        return (f"Get-Process -Id {pid} shows it and Stop-Process -Id {pid} "
                f"stops it (PowerShell)")
    return f"ps -p {pid} shows it and kill {pid} stops it"
