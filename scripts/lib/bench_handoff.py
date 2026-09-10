#!/usr/bin/env python3
"""Private, planned benchmark launches under retained supervisor GPU ownership.

This is a cooperative local protocol, not same-user authentication. Only the
selected executable receives the GPU capability; Cargo and ordinary commands do
not. The outer bench-locks process retains its copies through group cleanup.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import select
import selectors
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

PLAN_ENV = "LATTICE_GPU_HANDOFF_PLAN"
PLAN_DIGEST_ENV = "LATTICE_GPU_HANDOFF_PLAN_SHA256"
BROKER_ENV = "LATTICE_GPU_HANDOFF_BROKER"
BROKER_TOKEN_ENV = "LATTICE_GPU_HANDOFF_BROKER_TOKEN"
TARGET_ENVS = (
    "LATTICE_GPU_HANDOFF_FD",
    "LATTICE_GPU_HANDOFF_CONTROL",
    "LATTICE_GPU_HANDOFF_TOKEN",
)
REFUSAL_EXIT = 2
MAX_PLAN_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 65536
MAX_READY_BYTES = 4096
OUTPUT_DRAIN_TIMEOUT = 1.0


class HandoffError(RuntimeError):
    """Admission or its measured-phase witness could not be verified."""


def write_frozen_plan(repo: Path, plan: dict) -> tuple[Path, str]:
    """Persist one pre-lock admission result for the lock-wrapper re-exec."""
    directory = repo / ".cache" / "bench-supervision"
    directory.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="gpu-plan-", dir=directory))
    path = private / "plan.json"
    data = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    try:
        if len(data) > MAX_PLAN_BYTES:
            raise HandoffError("GPU admission plan is too large")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        private.rmdir()
        raise
    return path, hashlib.sha256(data).hexdigest()


def read_frozen_plan(
    repo: Path, mode: str | None = None, command: list[str] | None = None
) -> dict:
    """Read the exact private pre-lock plan without resolving display refs."""
    raw = os.environ.get(PLAN_ENV)
    digest = os.environ.get(PLAN_DIGEST_ENV)
    if not raw or not digest:
        raise HandoffError("GPU handoff has a missing frozen-plan component")
    path = Path(raw)
    expected_parent = (repo / ".cache" / "bench-supervision").resolve()
    if (
        path.name != "plan.json"
        or not path.parent.name.startswith("gpu-plan-")
        or path.parent.parent.resolve() != expected_parent
    ):
        raise HandoffError("GPU admission plan is outside its private directory")
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise HandoffError("GPU admission plan directory is not private")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            raise HandoffError("GPU admission plan is not a private regular file")
        data = stream.read(MAX_PLAN_BYTES + 1)
    if len(data) > MAX_PLAN_BYTES or hashlib.sha256(data).hexdigest() != digest:
        raise HandoffError("GPU admission plan changed after preflight")
    try:
        plan = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise HandoffError("GPU admission plan is not valid JSON") from exc
    if (
        not isinstance(plan, dict)
        or plan.get("version") != 1
        or plan.get("mode") not in ("compare", "command")
        or plan.get("repo") != str(repo)
        or not isinstance(plan.get("entries"), list)
        or not plan["entries"]
        or not isinstance(plan.get("environment"), dict)
        or not isinstance(plan.get("command"), list)
    ):
        raise HandoffError("GPU admission plan has an invalid schema")
    expected_ids = (
        ["base1", "head1", "head2", "base2"]
        if plan["mode"] == "compare"
        else ["command"]
    )
    if [
        entry.get("id") if isinstance(entry, dict) else None
        for entry in plan["entries"]
    ] != expected_ids:
        raise HandoffError("GPU admission plan has invalid invocation order")
    if mode is not None and plan["mode"] != mode:
        raise HandoffError("GPU admission mode changed after preflight")
    if command is not None and plan["command"] != command:
        raise HandoffError("GPU admission command changed after preflight")
    return plan


def remove_frozen_plan(repo: Path) -> None:
    """Remove only a validated plan file and its now-empty private directory."""
    read_frozen_plan(repo)
    path = Path(os.environ[PLAN_ENV])
    path.unlink()
    path.parent.rmdir()


def _receive(sock: socket.socket, maximum: int, deadline: float) -> dict:
    data = bytearray()
    while b"\n" not in data:
        if len(data) >= maximum:
            raise HandoffError("GPU handoff frame exceeds its size limit")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HandoffError("GPU handoff frame deadline expired")
        sock.settimeout(remaining)
        chunk = sock.recv(min(1024, maximum - len(data)))
        if not chunk:
            raise HandoffError("GPU handoff peer disconnected before its frame")
        data.extend(chunk)
    line, trailing = bytes(data).split(b"\n", 1)
    if trailing:
        raise HandoffError("GPU handoff sent more than one request frame")
    try:
        result = json.loads(line)
    except (ValueError, UnicodeError) as exc:
        raise HandoffError("GPU handoff frame is not valid JSON") from exc
    if not isinstance(result, dict):
        raise HandoffError("GPU handoff frame must be an object")
    return result


def _send(sock: socket.socket, value: dict) -> None:
    sock.sendall(json.dumps(value, separators=(",", ":")).encode() + b"\n")


def _check_helper_connection(client: socket.socket) -> None:
    readable, _, _ = select.select([client], [], [], 0)
    if not readable:
        return
    if not client.recv(1, socket.MSG_PEEK):
        raise HandoffError("measurement helper disconnected before completion")
    raise HandoffError("measurement helper sent another request before completion")


def _identity(path: Path) -> tuple[int, int]:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
        raise HandoffError("selected benchmark executable is not a regular executable")
    return metadata.st_dev, metadata.st_ino


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=2)


class HandoffService:
    """Serve ordered launches only; this interface cannot run arbitrary commands."""

    def __init__(
        self,
        plan: dict,
        gpu_fd: int,
        sample_locks: Callable[[], None],
        quiet: Callable[[str, str], tuple[bool, str]],
        *,
        validate_request: Callable[[dict, dict], Path] | None = None,
        ready_timeout: float = 120,
    ) -> None:
        self.plan = plan
        self.gpu_fd = gpu_fd
        self.sample_locks = sample_locks
        self.quiet = quiet
        if validate_request is None:
            from bench_admission import validate_measurement_request

            validate_request = validate_measurement_request
        self.validate_request = validate_request
        self.ready_timeout = ready_timeout
        self.token = secrets.token_hex(16)
        self.next_entry = 0
        self.failure: str | None = None
        self.stop = threading.Event()
        self.active_lock = threading.Lock()
        self.active: subprocess.Popen | None = None
        self.connections: set[socket.socket] = set()
        self.private = tempfile.TemporaryDirectory(prefix="lat-gpu-", dir="/tmp")
        self.broker_path = str(Path(self.private.name) / "broker")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.broker_path)
        self.listener.listen(1)
        self.listener.settimeout(0.1)
        receipt_dir = Path(plan["repo"]) / ".cache" / "bench-supervision"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        self.receipt_path = receipt_dir / f"gpu-phase-{self.token}.jsonl"
        fd = os.open(self.receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.receipt = os.fdopen(fd, "w")
        self.base_environment: dict[str, str] = {}

    def _output(self, client: socket.socket, output: bytes) -> None:
        for offset in range(0, len(output), 2048):
            _send(
                client,
                {"output": base64.b64encode(output[offset : offset + 2048]).decode()},
            )

    def _track(self, connection: socket.socket) -> None:
        with self.active_lock:
            if self.stop.is_set():
                connection.close()
                raise HandoffError("GPU handoff service is stopping")
            self.connections.add(connection)

    def _close(self, connection: socket.socket) -> None:
        with self.active_lock:
            self.connections.discard(connection)
        connection.close()

    def _record_phase(self, entry: dict, proc: subprocess.Popen, exe: Path, ok: bool, output: str) -> None:
        self.receipt.write(
            json.dumps(
                {
                    "protocol": 1,
                    "entry": entry["id"],
                    "revision": entry["revision"],
                    "target": entry["target"],
                    "pid": proc.pid,
                    "exe": str(exe),
                    "status": "ready" if ok else "refused",
                    "quiet": output,
                },
                sort_keys=True,
            )
            + "\n"
        )
        self.receipt.flush()

    def _launch(self, client: socket.socket, entry: dict, request: dict) -> int:
        exe = Path(self.validate_request(entry, request)).resolve(strict=True)
        identity = _identity(exe)
        self.sample_locks()
        token = secrets.token_hex(16)
        control_path = str(Path(self.private.name) / f"ready-{self.next_entry}")
        control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control.bind(control_path)
        control.listen(1)
        self._track(control)
        child_env = self.base_environment.copy()
        for name in (*TARGET_ENVS, BROKER_ENV, BROKER_TOKEN_ENV, PLAN_ENV, PLAN_DIGEST_ENV, "LATTICE_BENCH_LOCK_FDS", "LATTICE_BENCH_SUPERVISOR_FD"):
            child_env.pop(name, None)
        child_env.update(
            {
                TARGET_ENVS[0]: "0",
                TARGET_ENVS[1]: control_path,
                TARGET_ENVS[2]: token,
                "CRITERION_HOME": entry["criterion_home"],
            }
        )
        proc: subprocess.Popen | None = None
        try:
            with self.active_lock:
                if self.stop.is_set():
                    raise HandoffError("GPU handoff service is stopping")
                proc = subprocess.Popen(
                    [str(exe), *entry["argv"]],
                    cwd=entry["run_cwd"],
                    env=child_env,
                    stdin=self.gpu_fd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
                self.active = proc
            if proc.stdout is None:
                raise HandoffError("selected benchmark has no output stream")
            ready = False
            output_open = True
            drain_deadline: float | None = None
            deadline = time.monotonic() + self.ready_timeout
            with selectors.DefaultSelector() as selector:
                selector.register(control, selectors.EVENT_READ, "ready")
                selector.register(proc.stdout, selectors.EVENT_READ, "output")
                selector.register(client, selectors.EVENT_READ, "client")
                while output_open or proc.poll() is None:
                    if self.stop.is_set():
                        raise HandoffError("outer command ended during GPU measurement")
                    if proc.poll() is not None:
                        if not ready:
                            raise HandoffError("selected benchmark exited without measured-phase READY")
                        if drain_deadline is None:
                            drain_deadline = time.monotonic() + OUTPUT_DRAIN_TIMEOUT
                        if output_open and time.monotonic() >= drain_deadline:
                            raise HandoffError("selected benchmark exited with inherited output still open")
                    if drain_deadline is not None:
                        client.settimeout(max(0.01, drain_deadline - time.monotonic()))
                    if not ready and time.monotonic() >= deadline:
                        raise HandoffError("selected benchmark did not reach measured-phase READY")
                    for key, _ in selector.select(timeout=0.1):
                        if key.data == "client":
                            _check_helper_connection(client)
                        elif key.data == "output":
                            chunk = os.read(proc.stdout.fileno(), 2048)
                            if chunk:
                                self._output(client, chunk)
                            else:
                                selector.unregister(proc.stdout)
                                output_open = False
                        else:
                            connection, _ = control.accept()
                            self._track(connection)
                            try:
                                message = _receive(connection, MAX_READY_BYTES, deadline)
                                if (
                                    set(message) != {"protocol", "token", "pid", "exe"}
                                    or type(message["protocol"]) is not int
                                    or message["protocol"] != 1
                                    or message["token"] != token
                                    or type(message["pid"]) is not int
                                    or message["pid"] != proc.pid
                                    or not isinstance(message["exe"], str)
                                    or not Path(message["exe"]).is_absolute()
                                    or _identity(Path(message["exe"])) != identity
                                    or _identity(exe) != identity
                                ):
                                    raise HandoffError("selected benchmark READY does not match its invocation")
                                _check_helper_connection(client)
                                self.sample_locks()
                                ok, output = self.quiet(
                                    f"{entry['id']}:{entry['target']}: measured guard",
                                    self.plan["mode"],
                                )
                                self._record_phase(entry, proc, exe, ok, output)
                                self._output(client, output.encode())
                                if not ok:
                                    raise HandoffError("measured-phase quiet certification failed")
                                if proc.poll() is not None or self.stop.is_set():
                                    raise HandoffError("selected benchmark exited before acknowledgement")
                                if time.monotonic() >= deadline:
                                    raise HandoffError("measured-phase acknowledgement deadline expired")
                                _check_helper_connection(client)
                                self.sample_locks()
                                _send(connection, {"protocol": 1, "token": token, "status": "ready"})
                                ready = True
                            finally:
                                self._close(connection)
                            selector.unregister(control)
                            self._close(control)
                    if proc.poll() is not None and not ready:
                        raise HandoffError("selected benchmark exited without measured-phase READY")
            returncode = proc.wait()
            if not ready:
                raise HandoffError("selected benchmark produced no measured-phase witness")
            self.sample_locks()
            return returncode if returncode >= 0 else 128 - returncode
        finally:
            _terminate(proc)
            if proc is not None and proc.stdout is not None:
                proc.stdout.close()
            with self.active_lock:
                self.active = None
            self._close(control)
            Path(control_path).unlink(missing_ok=True)

    def _serve(self) -> None:
        while not self.stop.is_set():
            try:
                client, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if not self.stop.is_set():
                    self.failure = str(exc)
                return
            try:
                self._track(client)
                client.settimeout(self.ready_timeout)
                request = _receive(
                    client, MAX_REQUEST_BYTES, time.monotonic() + self.ready_timeout
                )
                if (
                    type(request.get("protocol")) is not int
                    or request["protocol"] != 1
                    or request.get("token") != self.token
                    or self.next_entry >= len(self.plan["entries"])
                ):
                    raise HandoffError("unrecognized or replayed GPU measurement request")
                entry = self.plan["entries"][self.next_entry]
                if request.get("entry") != entry["id"]:
                    raise HandoffError("GPU measurement request is out of plan order")
                status = self._launch(client, entry, request)
                if status != 0:
                    raise HandoffError(f"selected benchmark failed with exit {status}")
                self.next_entry += 1
                _send(client, {"status": status})
            except Exception as exc:
                self.failure = str(exc)
                try:
                    self._output(client, f"bench-supervision: {exc}; refusing to measure\n".encode())
                    _send(client, {"status": REFUSAL_EXIT})
                except OSError:
                    pass
                return
            finally:
                self._close(client)

    def run(self, command: list[str], env: dict[str, str], pass_fds: tuple[int, ...]) -> int:
        """Run the outer body while the private broker services its planned launches."""
        self.base_environment = env.copy()
        child_env = env.copy()
        child_env[BROKER_ENV] = self.broker_path
        child_env[BROKER_TOKEN_ENV] = self.token
        print(f"bench-supervision: measured-phase receipt: {self.receipt_path}", file=sys.stderr)
        worker = threading.Thread(target=self._serve, name="gpu-handoff", daemon=True)
        outer: subprocess.Popen | None = None
        worker.start()
        try:
            outer = subprocess.Popen(command, env=child_env, pass_fds=pass_fds)
            while outer.poll() is None and self.failure is None:
                time.sleep(0.02)
            if self.failure is not None:
                print(f"bench-supervision: {self.failure}; refusing to certify", file=sys.stderr)
                return REFUSAL_EXIT
            if outer.returncode != 0:
                return outer.returncode
            if self.next_entry != len(self.plan["entries"]):
                print("bench-supervision: planned GPU measurements did not all complete", file=sys.stderr)
                return REFUSAL_EXIT
            return 0
        finally:
            self.stop.set()
            _terminate(outer)
            self.listener.close()
            with self.active_lock:
                active = self.active
                connections = list(self.connections)
            for connection in connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            _terminate(active)
            worker.join(timeout=2)
            self.receipt.close()
            self.private.cleanup()
