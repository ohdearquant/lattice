#!/usr/bin/env python3
"""Exercise decode adapters through the real supervisor using local stubs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class _OllamaServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, fail_unload: bool = False):
        super().__init__(address, handler)
        self.loaded = False
        self.fail_unload = fail_unload
        self.requests: list[dict] = []


class _OllamaHandler(BaseHTTPRequestHandler):
    server: _OllamaServer

    def log_message(self, _format, *_args):
        return

    def _reply(self, data: dict):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._reply({"models": [{"name": "qwen3.5:0.8b"}]})
            return
        if self.path == "/api/ps":
            self.server.requests.append({"path": self.path})
            models = [{"name": "qwen3.5:0.8b"}] if self.server.loaded else []
            self._reply({"models": models})
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size))
        self.server.requests.append(body)
        if body.get("prompt") == "" and body.get("keep_alive") == 0:
            if not self.server.fail_unload:
                self.server.loaded = False
            self._reply({"done": True})
            return
        self.server.loaded = True
        count = body.get("options", {}).get("num_predict", 4)
        self._reply(
            {
                "done": True,
                "eval_count": count,
                "eval_duration": 2_000_000,
                "total_duration": 4_000_000,
                "load_duration": 1_000_000,
                "prompt_eval_duration": 1_000_000,
                "prompt_eval_count": 8,
            }
        )


class DecodeSupervisorFixture:
    """Small committed tree so the real admission code can require a clean HEAD."""

    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="decode-supervision-")
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.home = Path(self.temporary.name) / "home"
        self.home.mkdir()
        self.stubs = Path(self.temporary.name) / "stubs"
        self.stubs.mkdir()
        self.bin = Path(self.temporary.name) / "bin"
        self.bin.mkdir()
        self.bench_lock = Path(self.temporary.name) / "bench.lock"
        self.gpu_lock = Path(self.temporary.name) / "gpu.lock"
        self.events = Path(self.temporary.name) / "events.jsonl"
        self.publish_events = Path(self.temporary.name) / "publish.jsonl"
        self.status_file = self.root / ".cache" / "fixture-status"
        self._copy_inputs()
        self._write_stubs()
        self._commit_fixture()
        model = self.home / ".lattice" / "models" / "qwen3.5-0.8b"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}\n")

    def close(self):
        self.temporary.cleanup()

    def clean(self):
        subprocess.run(["git", "-C", str(self.root), "reset", "--hard", "HEAD"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.root), "clean", "-fdx"], check=True, capture_output=True)
        self.events.unlink(missing_ok=True)
        self.publish_events.unlink(missing_ok=True)

    def _copy(self, relative: str):
        source = REPO / relative
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def _copy_inputs(self):
        for relative in (
            ".gitignore",
            "Cargo.toml",
            "scripts/bench-command.sh",
            "scripts/bench_decode_harness.py",
            "scripts/bench_gate_math.py",
            "scripts/bench_decode_profiles.toml",
            "scripts/bench_decode_adapters_agentic.py",
            "scripts/bench_decode_adapters_apples_to_apples.py",
            "scripts/bench_decode_adapters_q4_apples.py",
            "scripts/lib/bench-python.sh",
            "scripts/lib/bench_supervision.py",
            "scripts/lib/bench_admission.py",
            "scripts/lib/bench_handoff.py",
            "scripts/lib/bench-locks.py",
            "scripts/lib/quiet-probe.py",
            "crates/inference/Cargo.toml",
            "crates/inference/src/bin/bench_decode_ab.rs",
        ):
            self._copy(relative)

        locks = self.root / "scripts/lib/bench-locks.py"
        source = locks.read_text()
        source = "\n".join(
            f"BENCH_WINDOW = {str(self.bench_lock)!r}" if line.startswith("BENCH_WINDOW =")
            else f"GPU_LOCK = {str(self.gpu_lock)!r}" if line.startswith("GPU_LOCK =")
            else line
            for line in source.splitlines()
        ) + "\n"
        locks.write_text(source)

        probe = self.root / "scripts/lib/quiet-probe.py"
        probe.write_text(
            "import os, sys\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "if os.environ.get('FIXTURE_FAIL_BEFORE') == '1' and label.endswith(': before'):\n"
            "    raise SystemExit(1)\n"
            "if os.environ.get('FIXTURE_FAIL_AFTER') == '1' and label.endswith(': after'):\n"
            "    raise SystemExit(1)\n"
            "if os.environ.get('FIXTURE_FAIL_AFTER_LATTICE') == '1' and label.startswith('decode-lattice-') and label.endswith(': after'):\n"
            "    raise SystemExit(1)\n"
            "print('QUIET fixture')\n"
        )

        agentic = self.root / "scripts/bench_decode_adapters_agentic.py"
        source = agentic.read_text().replace(
            "SWEEP_CONTEXTS = (1000, 2000, 4000)", "SWEEP_CONTEXTS = (8, 16, 24)"
        )
        agentic.write_text(source)

        inference_manifest = self.root / "crates/inference/Cargo.toml"
        self.manifest = tomllib.loads(inference_manifest.read_text())

    def _write_stubs(self):
        (self.stubs / "mlx_lm").mkdir()
        (self.stubs / "mlx").mkdir()
        (self.stubs / "fixture_events.py").write_text(
            "import contextlib, fcntl, json, os, subprocess, time\n"
            "def _held(path):\n"
            "    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)\n"
            "    try:\n"
            "        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "        except BlockingIOError: return True\n"
            "        else: fcntl.flock(fd, fcntl.LOCK_UN); return False\n"
            "    finally: os.close(fd)\n"
            "def record(engine, kind, phase):\n"
            "    held = all(_held(os.environ[name]) for name in ('FIXTURE_BENCH_LOCK', 'FIXTURE_GPU_LOCK'))\n"
            "    clean = None\n"
            "    if phase == 'start' and os.environ.get('FIXTURE_CHECK_CLEAN') == '1':\n"
            "        status = subprocess.run(['git', '-C', os.environ['FIXTURE_REPO'], 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout\n"
            "        clean = not status.strip()\n"
            "    row = dict(engine=engine, kind=kind, phase=phase, time_ns=time.monotonic_ns(), locks_held=held, clean=clean)\n"
            "    fd = os.open(os.environ['FIXTURE_EVENTS'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)\n"
            "    try: os.write(fd, (json.dumps(row, sort_keys=True) + '\\n').encode())\n"
            "    finally: os.close(fd)\n"
            "@contextlib.contextmanager\n"
            "def event(engine, kind):\n"
            "    record(engine, kind, 'start')\n"
            "    try: yield\n"
            "    finally: record(engine, kind, 'end')\n"
        )

        (self.stubs / "sitecustomize.py").write_text(
            "import json, os, sys, time, urllib.parse, urllib.request\n"
            "sys.platform = 'darwin'\n"
            "from fixture_events import event, record\n"
            "_urlopen = urllib.request.urlopen\n"
            "class _Response:\n"
            "    def __init__(self, response, scope): self.response, self.scope = response, scope\n"
            "    def __enter__(self): self.response.__enter__(); return self\n"
            "    def __exit__(self, *args):\n"
            "        try: return self.response.__exit__(*args)\n"
            "        finally: self.scope.__exit__(*args)\n"
            "    def read(self, *args): return self.response.read(*args)\n"
            "    def __getattr__(self, name): return getattr(self.response, name)\n"
            "def _traced_urlopen(url, *args, **kwargs):\n"
            "    value = url.full_url if isinstance(url, urllib.request.Request) else str(url)\n"
            "    parsed = urllib.parse.urlsplit(value)\n"
            "    if parsed.path == '/api/tags': return _urlopen(url, *args, **kwargs)\n"
            "    if parsed.path not in ('/api/generate', '/api/ps'): return _urlopen(url, *args, **kwargs)\n"
            "    if parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:\n"
            "        record('ollama', 'nonloopback_attempt', 'start')\n"
            "        raise RuntimeError('fixture blocked non-loopback network request')\n"
            "    payload = {}\n"
            "    if isinstance(url, urllib.request.Request) and url.data:\n"
            "        try: payload = json.loads(url.data)\n"
            "        except (TypeError, ValueError): pass\n"
            "    kind = 'loaded_check' if parsed.path == '/api/ps' else ('unload' if payload.get('prompt') == '' else 'generate')\n"
            "    scope = event('ollama', kind); scope.__enter__()\n"
            "    try: return _Response(_urlopen(url, *args, **kwargs), scope)\n"
            "    except BaseException as exc:\n"
            "        scope.__exit__(type(exc), exc, exc.__traceback__); raise\n"
            "urllib.request.urlopen = _traced_urlopen\n"
            "_copy2 = __import__('shutil').copy2\n"
            "def _traced_copy2(src, dst, *args, **kwargs):\n"
            "    if '--profile' in sys.argv and 'agentic' in sys.argv:\n"
            "        record('publication', str(dst), 'start')\n"
            "        try: return _copy2(src, dst, *args, **kwargs)\n"
            "        finally: record('publication', str(dst), 'end')\n"
            "    return _copy2(src, dst, *args, **kwargs)\n"
            "__import__('shutil').copy2 = _traced_copy2\n"
            "from pathlib import Path\n"
            "_replace = Path.replace\n"
            "def _traced_replace(self, target):\n"
            "    result = _replace(self, target)\n"
            "    if str(target).endswith('-result.json'):\n"
            "        record('worker', 'result_written', 'end')\n"
            "    return result\n"
            "Path.replace = _traced_replace\n"
        )

        (self.stubs / "mlx_lm/__init__.py").write_text(
            "from fixture_events import event\n"
            "class Tokenizer:\n"
            "    def encode(self, text): return text.split()\n"
            "class Model:\n"
            "    def parameters(self): return []\n"
            "from . import utils\n"
            "def load(model_id):\n"
            "    with event('mlx', 'load'): return Model(), Tokenizer()\n"
            "def generate(model, tokenizer, **kwargs):\n"
            "    with event('mlx', 'generate'): return 'stub output'\n"
        )
        (self.stubs / "mlx_lm/utils.py").write_text(
            "from fixture_events import event\n"
            "from . import Tokenizer\n"
            "def load_tokenizer(model_id):\n"
            "    with event('mlx', 'tokenizer_prepare'): return Tokenizer()\n"
        )
        (self.stubs / "mlx_lm/sample_utils.py").write_text(
            "def make_sampler(temp=0.0): return object()\n"
        )
        (self.stubs / "mlx/__init__.py").write_text("")
        (self.stubs / "mlx/core.py").write_text(
            "from fixture_events import event\n"
            "def array(value): return value\n"
            "def eval(value):\n"
            "    with event('mlx', 'eval'): return None\n"
        )
        (self.stubs / "mlx/nn.py").write_text(
            "from fixture_events import event\n"
            "def quantize(model, **kwargs):\n"
            "    with event('mlx', 'quantize'): return None\n"
        )

        compiler = shutil.which("cc") or shutil.which("clang")
        if compiler is None:
            raise RuntimeError("the supervised fixture requires a C compiler for its stub binary")
        cargo_source = """#!PYTHON
import json, os, pathlib, subprocess, sys, tomllib
sys.path.insert(0, __LIB__)
import bench_admission
args = sys.argv[1:]
target = args[args.index('--bin') + 1]
features = args[args.index('--features') + 1]
target_dir = pathlib.Path(args[args.index('--target-dir') + 1])
repo = pathlib.Path(os.environ['FIXTURE_REPO'])
status = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
fd = os.open(os.environ['FIXTURE_EVENTS'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(fd, (json.dumps(dict(engine='build', kind='admission', phase='start', clean=not status.strip())) + '\\n').encode())
os.close(fd)
manifest = tomllib.loads((repo / 'crates/inference/Cargo.toml').read_text())
root = tomllib.loads((repo / 'Cargo.toml').read_text())
package = manifest['package']
version = package['version']
if isinstance(version, dict):
    version = root['workspace']['package']['version']
source = repo / 'crates/inference/src/bin' / (target + '.rs')
target_dir.mkdir(parents=True, exist_ok=True)
exe = target_dir / 'release' / target
exe.parent.mkdir(parents=True, exist_ok=True)
c_source = exe.with_suffix('.c')
c_source.write_text(os.environ['FIXTURE_STUB_C'])
subprocess.run([__COMPILER__, '-O', '-o', str(exe), str(c_source)], check=True)
feature_set = bench_admission.feature_closure(manifest, features)
package_id = f"path+{(repo / 'crates/inference').resolve().as_uri()}#lattice-inference@{version}"
artifact = {'reason':'compiler-artifact','target':{'name':target,'kind':['bin'],'src_path':str(source)},'features':feature_set,'package_id':package_id,'executable':str(exe)}
print(json.dumps(artifact))
"""
        cargo_source = cargo_source.replace("#!PYTHON", f"#!{sys.executable}")
        cargo_source = cargo_source.replace("__LIB__", repr(str(self.root / "scripts/lib")))
        cargo_source = cargo_source.replace("__COMPILER__", repr(compiler))
        cargo = self.bin / "cargo"
        cargo.write_text(cargo_source)
        cargo.chmod(0o755)
        (self.bin / "ollama").write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "if sys.argv[1:2] == ['list']: print('NAME\\nqwen3.5:0.8b')\n"
            "elif sys.argv[1:2] == ['pull']: print('success')\n"
            "else: raise SystemExit('fixture Ollama CLI does not start a server')\n"
        )
        (self.bin / "ollama").chmod(0o755)
        self.stub_c = r'''#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

static int held(const char *path) {
    int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) return 0;
    int rc = flock(fd, LOCK_EX | LOCK_NB);
    if (rc == 0) flock(fd, LOCK_UN);
    close(fd);
    return rc != 0;
}

static void log_event(const char *kind, const char *phase) {
    const char *events = getenv("FIXTURE_EVENTS");
    int fd = open(events, O_WRONLY | O_CREAT | O_APPEND, 0600);
    struct timespec now;
    /* Stamp with the clock Python's time.monotonic_ns() reads: on macOS that is
       mach absolute time (CLOCK_UPTIME_RAW), which differs from CLOCK_MONOTONIC. */
#if defined(__APPLE__)
    clock_gettime(CLOCK_UPTIME_RAW, &now);
#else
    clock_gettime(CLOCK_MONOTONIC, &now);
#endif
    long long stamp = (long long)now.tv_sec * 1000000000LL + now.tv_nsec;
    char line[512];
    int clean = 1;
    if (getenv("FIXTURE_CHECK_CLEAN")) {
        char command[PATH_MAX + 64];
        snprintf(command, sizeof(command), "git -C '%s' status --porcelain > '%s'", getenv("FIXTURE_REPO"), getenv("FIXTURE_STATUS_FILE"));
        system(command);
        FILE *status = fopen(getenv("FIXTURE_STATUS_FILE"), "r");
        if (status && fgetc(status) != EOF) clean = 0;
        if (status) fclose(status);
    }
    snprintf(line, sizeof(line), "{\"engine\":\"lattice\",\"kind\":\"%s\",\"phase\":\"%s\",\"time_ns\":%lld,\"locks_held\":%s,\"clean\":%s}\n",
        kind, phase, stamp, held(getenv("FIXTURE_BENCH_LOCK")) && held(getenv("FIXTURE_GPU_LOCK")) ? "true" : "false", clean ? "true" : "false");
    write(fd, line, strlen(line));
    close(fd);
}

int main(int argc, char **argv) {
    if (getenv("FIXTURE_CLOCK_PROBE")) {
        log_event("clock_probe", "start");
        return 0;
    }
    const char *control = getenv("LATTICE_GPU_HANDOFF_CONTROL");
    const char *token = getenv("LATTICE_GPU_HANDOFF_TOKEN");
    char exe[PATH_MAX];
    if (!realpath(argv[0], exe)) return 20;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    strncpy(address.sun_path, control, sizeof(address.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) return 21;
    char request[PATH_MAX + 256];
    snprintf(request, sizeof(request), "{\"protocol\":1,\"token\":\"%s\",\"pid\":%d,\"exe\":\"%s\"}\n", token, getpid(), exe);
    write(fd, request, strlen(request));
    char reply[256] = {0};
    if (read(fd, reply, sizeof(reply) - 1) <= 0 || !strstr(reply, "ready")) return 22;
    log_event("decode", "start");
    usleep(10000);
    const char *n = getenv("BENCH_N");
    const char *prompt = getenv("BENCH_PROMPT_TOKENS");
    printf("RESULT n_req=%s completion=%s total_ms=1.0\n", n ? n : "32", n ? n : "32");
    if (prompt) printf("[bench] prompt_tokens=%s\n", prompt);
    fflush(stdout);
    log_event("printed_result", "end");
    log_event("decode", "end");
    close(fd);
    return 0;
}
'''
        (self.root / "scripts/fixture_driver.py").write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "import bench_decode_harness as harness\n"
            "from bench_decode_adapters_q4_apples import MlxAdapter\n"
            "_, profiles = harness.load_profiles_file(Path(__file__).parent / 'bench_decode_profiles.toml')\n"
            "result = harness.run_profile(profiles['q4_apples'], {'mlx': MlxAdapter()}, allow_missing_engine=True, supervised_children=True)\n"
            "Path(os.environ['FIXTURE_DRIVER_RESULT']).write_text(json.dumps([row.to_dict() for row in result.observations]))\n"
        )
        (self.root / "scripts/fixture_ollama_driver.py").write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "import bench_decode_harness as harness\n"
            "import bench_decode_adapters_agentic as agentic\n"
            "adapter = agentic.OllamaAdapter(os.environ['LATTICE_BENCH_OLLAMA_URL'])\n"
            "if os.environ.get('FIXTURE_NON_LOOPBACK') == '1': adapter.base_url = 'http://203.0.113.1:11434'\n"
            "profile = agentic.configure_profile(agentic._default_profile(), ctx=8, runs=1, padded_prompt='fixture padded prompt')\n"
            "result = harness.run_profile(profile, {'ollama': adapter}, allow_missing_engine=True, supervised_children=True)\n"
            "Path(os.environ['FIXTURE_DRIVER_RESULT']).write_text(json.dumps([row.to_dict() for row in result.observations]))\n"
        )

    def _commit_fixture(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(self.root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture source"],
            check=True,
        )

    def env(self, updates: dict[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        for name in (
            "LATTICE_BENCH_LOCK_STATUS", "LATTICE_BENCH_LOCK_FDS", "LATTICE_BENCH_SUPERVISOR_FD",
            "LATTICE_GPU_HANDOFF_CONTROL", "LATTICE_GPU_HANDOFF_FD", "LATTICE_GPU_HANDOFF_TOKEN",
        ):
            env.pop(name, None)
        env.update(
            {
                "PATH": f"{self.bin}:{os.environ.get('PATH', '')}",
                "PYTHONPATH": f"{self.stubs}:{self.root / 'scripts'}:{os.environ.get('PYTHONPATH', '')}",
                "PYTHON_BIN": sys.executable,
                "HOME": str(self.home),
                "FIXTURE_REPO": str(self.root),
                "FIXTURE_EVENTS": str(self.events),
                "FIXTURE_PUBLISH_EVENTS": str(self.publish_events),
                "FIXTURE_STATUS_FILE": str(self.status_file),
                "FIXTURE_BENCH_LOCK": str(self.bench_lock),
                "FIXTURE_GPU_LOCK": str(self.gpu_lock),
                "FIXTURE_STUB_C": self.stub_c,
                "FIXTURE_CHECK_CLEAN": "1",
                "LATTICE_BENCH_OLLAMA_URL": "http://127.0.0.1:11434",
                "LATTICE_BENCH_HARDWARE_ID": f"fixture-{time.time_ns()}",
            }
        )
        if updates:
            env.update(updates)
        return env

    def run(self, script: str, args: list[str], *, updates: dict[str, str] | None = None):
        return subprocess.run(
            [sys.executable, str(self.root / script), *args],
            cwd=self.root,
            env=self.env(updates),
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )

    def read_events(self) -> list[dict]:
        if not self.events.exists():
            return []
        return [json.loads(line) for line in self.events.read_text().splitlines() if line]


class BenchDecodeSupervisedEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = DecodeSupervisorFixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.close()

    def setUp(self):
        self.fixture.clean()

    def _serve_ollama(self, *, fail_unload: bool = False):
        server = _OllamaServer(("127.0.0.1", 0), _OllamaHandler, fail_unload=fail_unload)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _stop_ollama(self, server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    def test_three_context_sweep_holds_locks_and_publishes_after_final_child(self):
        server, thread = self._serve_ollama()
        try:
            env = self.fixture.env(
                {
                    "LATTICE_BENCH_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
                    "FIXTURE_CHECK_CLEAN": "1",
                }
            )
            result = subprocess.run(
                [sys.executable, str(self.fixture.root / "scripts/bench_decode_harness.py"), "run", "--profile", "agentic", "--sweep", "--runs", "1", "--allow-missing-engine"],
                cwd=self.fixture.root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )
        finally:
            self._stop_ollama(server, thread)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows_path = self.fixture.root / "docs/bench_results/agentic_sweep.json"
        self.assertTrue(rows_path.is_file())
        rows = json.loads(rows_path.read_text())
        self.assertEqual(len(rows), 9)
        for context in (8, 16, 24):
            context_rows = json.loads(
                (self.fixture.root / f"docs/bench_results/agentic_{context}tok.json").read_text()
            )
            self.assertEqual({row["engine"] for row in context_rows}, {"lattice", "ollama", "mlx"})
            self.assertEqual(next(row for row in context_rows if row["engine"] == "lattice")["context"], context)
        ollama_rows = [row for row in rows if row["engine"] == "ollama"]
        self.assertEqual(len(ollama_rows), 3)
        self.assertTrue(all(row["scope"].startswith("external server;") for row in ollama_rows))
        self.assertTrue(all(row["unload_confirmed"] is True for row in ollama_rows))

        events = self.fixture.read_events()
        engine_events = [row for row in events if row["engine"] in {"lattice", "mlx", "ollama"}]
        self.assertTrue(engine_events)
        self.assertTrue(all(row["locks_held"] for row in engine_events), engine_events)
        self.assertTrue(all(row["clean"] is True for row in engine_events if row["phase"] == "start"), engine_events)
        self.assertEqual(sum(row["engine"] == "lattice" and row["phase"] == "start" for row in events), 6)
        self.assertEqual(sum(row["engine"] == "mlx" and row["kind"] == "tokenizer_prepare" and row["phase"] == "start" for row in events), 3)
        self.assertEqual(sum(row["engine"] == "ollama" and row["kind"] == "unload" and row["phase"] == "start" for row in events), 3)

        pending: dict[tuple[str, str], list[int]] = {}
        intervals = []
        for row in sorted(events, key=lambda item: item.get("time_ns", 0)):
            if row["engine"] not in {"lattice", "mlx", "ollama"}:
                continue
            key = (row["engine"], row["kind"])
            if row["phase"] == "start":
                pending.setdefault(key, []).append(row["time_ns"])
            elif pending.get(key):
                intervals.append((pending[key].pop(0), row["time_ns"]))
        intervals.sort()
        self.assertTrue(all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])), intervals)

        publication = [row for row in events if row["engine"] == "publication" and row["phase"] == "start"]
        last_engine_end = max(row["time_ns"] for row in engine_events if row["phase"] == "end")
        self.assertTrue(publication)
        self.assertGreater(min(row["time_ns"] for row in publication), last_engine_end)

    def test_stub_event_clock_matches_python_monotonic_clock(self):
        work = Path(self.fixture.temporary.name) / "clock-probe"
        work.mkdir(exist_ok=True)
        source = work / "stub.c"
        executable = work / "stub"
        events = work / "events.jsonl"
        events.unlink(missing_ok=True)
        source.write_text(self.fixture.stub_c)
        compiler = shutil.which("cc") or shutil.which("clang")
        subprocess.run([compiler, "-O", "-o", str(executable), str(source)], check=True)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "FIXTURE_CLOCK_PROBE": "1",
            "FIXTURE_EVENTS": str(events),
            "FIXTURE_BENCH_LOCK": str(self.fixture.bench_lock),
            "FIXTURE_GPU_LOCK": str(self.fixture.gpu_lock),
        }
        before = time.monotonic_ns()
        result = subprocess.run([str(executable)], env=env, capture_output=True, text=True, check=False, timeout=30)
        after = time.monotonic_ns()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in events.read_text().splitlines() if line]
        self.assertEqual([row["kind"] for row in rows], ["clock_probe"])
        stamp = rows[0]["time_ns"]
        self.assertLessEqual(before, stamp, (before, stamp, after))
        self.assertLessEqual(stamp, after, (before, stamp, after))

    def test_before_probe_refusal_starts_no_engine(self):
        server, thread = self._serve_ollama()
        try:
            result = subprocess.run(
                [sys.executable, str(self.fixture.root / "scripts/bench_decode_harness.py"), "run", "--profile", "agentic", "--ctx", "8", "--runs", "1", "--allow-missing-engine"],
                cwd=self.fixture.root,
                env=self.fixture.env(
                    {
                        "LATTICE_BENCH_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
                        "FIXTURE_FAIL_BEFORE": "1",
                    }
                ),
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        finally:
            self._stop_ollama(server, thread)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("quiet", result.stderr.lower(), result.stdout + result.stderr)
        self.assertEqual(self.fixture.read_events(), [])
        self.assertFalse((self.fixture.root / "docs/bench_results/agentic_8tok.json").exists())

    def test_supervisor_after_probe_discards_printed_lattice_rows(self):
        server, thread = self._serve_ollama()
        try:
            result = subprocess.run(
                [sys.executable, str(self.fixture.root / "scripts/bench_decode_harness.py"), "run", "--profile", "agentic", "--ctx", "8", "--runs", "1", "--allow-missing-engine"],
                cwd=self.fixture.root,
                env=self.fixture.env(
                    {
                        "LATTICE_BENCH_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
                        "FIXTURE_FAIL_AFTER_LATTICE": "1",
                    }
                ),
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        finally:
            self._stop_ollama(server, thread)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(
            any(row["kind"] == "printed_result" for row in self.fixture.read_events()),
            result.stderr,
        )
        self.assertFalse((self.fixture.root / "docs/bench_results/agentic_8tok.json").exists())
        self.assertTrue(any(row["engine"] == "lattice" and row["phase"] == "end" for row in self.fixture.read_events()))

    def test_child_result_is_rejected_when_supervisor_returns_two(self):
        result = self.fixture.run(
            "scripts/fixture_driver.py",
            [],
            updates={"FIXTURE_FAIL_AFTER": "1", "FIXTURE_DRIVER_RESULT": str(self.fixture.root / ".cache/driver.json")},
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(any(row["kind"] == "result_written" for row in self.fixture.read_events()))
        self.assertFalse((self.fixture.root / ".cache/driver.json").exists())
        self.assertFalse(any(path.name.startswith("decode-worker-") for path in (self.fixture.root / ".cache").glob("*")))

    def test_ollama_child_rejects_non_loopback_before_request(self):
        server, thread = self._serve_ollama()
        try:
            result = self.fixture.run(
                "scripts/fixture_ollama_driver.py",
                [],
                updates={
                    "LATTICE_BENCH_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
                    "FIXTURE_NON_LOOPBACK": "1",
                    "FIXTURE_DRIVER_RESULT": str(self.fixture.root / ".cache/ollama.json"),
                },
            )
        finally:
            self._stop_ollama(server, thread)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(server.requests, [])
        self.assertFalse(any(row["kind"] == "nonloopback_attempt" for row in self.fixture.read_events()))

    def test_ollama_child_loopback_control_makes_requests(self):
        server, thread = self._serve_ollama()
        try:
            result = self.fixture.run(
                "scripts/fixture_ollama_driver.py",
                [],
                updates={
                    "LATTICE_BENCH_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
                    "FIXTURE_DRIVER_RESULT": str(self.fixture.root / ".cache/ollama.json"),
                },
            )
        finally:
            self._stop_ollama(server, thread)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(server.requests)
        self.assertTrue(
            any("path" not in request and request.get("prompt") for request in server.requests),
            server.requests,
        )
        self.assertTrue(any(request.get("prompt") == "" for request in server.requests), server.requests)
        self.assertTrue(any(request.get("path") == "/api/ps" for request in server.requests), server.requests)
        self.assertTrue((self.fixture.root / ".cache/ollama.json").is_file())

    def test_ollama_unload_control_succeeds_when_model_unloads(self):
        server, thread = self._serve_ollama(fail_unload=False)
        try:
            result = self.fixture.run(
                "scripts/fixture_ollama_driver.py",
                [],
                updates={
                    "LATTICE_BENCH_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
                    "FIXTURE_DRIVER_RESULT": str(self.fixture.root / ".cache/ollama.json"),
                },
            )
        finally:
            self._stop_ollama(server, thread)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("unload/check failed", result.stderr)
        self.assertTrue((self.fixture.root / ".cache/ollama.json").is_file())
        self.assertTrue(any(request.get("prompt") == "" for request in server.requests), server.requests)
        self.assertTrue(any(request.get("path") == "/api/ps" for request in server.requests), server.requests)

    def test_ollama_unload_check_failure_is_a_failed_run(self):
        server, thread = self._serve_ollama(fail_unload=True)
        try:
            result = self.fixture.run(
                "scripts/fixture_ollama_driver.py",
                [],
                updates={
                    "LATTICE_BENCH_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
                    "FIXTURE_DRIVER_RESULT": str(self.fixture.root / ".cache/ollama.json"),
                },
            )
        finally:
            self._stop_ollama(server, thread)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("unload/check failed", result.stderr)
        self.assertTrue(any(request.get("prompt") == "" for request in server.requests))
        self.assertTrue(any(request.get("path") == "/api/ps" for request in server.requests))
        self.assertFalse((self.fixture.root / ".cache/ollama.json").exists())

    def test_mlx_load_quantize_warmup_and_measurements_live_in_child(self):
        result = self.fixture.run(
            "scripts/fixture_driver.py",
            [],
            updates={"FIXTURE_DRIVER_RESULT": str(self.fixture.root / ".cache/driver.json")},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        observations = json.loads((self.fixture.root / ".cache/driver.json").read_text())
        self.assertTrue(any(row["warmup"] for row in observations))
        self.assertTrue(any(not row["warmup"] for row in observations))
        self.assertTrue(all(row["elapsed_ns"] < 100_000_000 for row in observations))
        events = self.fixture.read_events()
        self.assertTrue(any(row["kind"] == "load" for row in events))
        self.assertTrue(any(row["kind"] == "quantize" for row in events))
        self.assertTrue(any(row["kind"] == "generate" and row["phase"] == "start" for row in events))
        self.assertTrue(all(row["locks_held"] for row in events if row["engine"] == "mlx"))


if __name__ == "__main__":
    unittest.main()
