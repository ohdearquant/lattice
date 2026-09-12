#!/bin/bash
# Run one phase under the caller's machine locks and admission checks.
set -euo pipefail
phase=${1:?expected phase}
: "${CARGO_TARGET_DIR:?expected relay target directory}"
mkdir -p w6-out
UV_CACHE_DIR=w6-out/uv-cache uv run --no-project python3 - "$phase" <<'PY'
import json, os, pathlib, subprocess, sys, time
phase = sys.argv[1]
out = pathlib.Path('w6-out')
bin_dir = pathlib.Path(os.environ['CARGO_TARGET_DIR']) / 'release'
model = pathlib.Path.home() / '.lattice/models/qwen3.5-0.8b'
def conditions():
    names = ['kern.memorystatus_vm_pressure_level', 'vm.loadavg', 'hw.model', 'hw.memsize', 'machdep.cpu.brand_string']
    return {name: subprocess.check_output(['sysctl', '-n', name], text=True).strip() for name in names}
receipt = {'phase': phase, 'start': conditions(), 'window': 'shared; informational timing only'}
start = time.monotonic()
def run(cmd, path):
    with open(path, 'wb') as log:
        result = subprocess.run([str(x) for x in cmd], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{cmd[0]} exited {result.returncode}; see {path}')
try:
    if phase in ('train-M', 'train-G'):
        family = 'memory' if phase == 'train-M' else 'gtd'
        run([bin_dir / 'train_grad_full', '--model-dir', model, '--data-dir', f'w6-input/adapter-{family}', '--seq-len', '256', '--steps', '96', '--max-train', '48', '--max-valid', '8', '--first-layer', '19', '--rank', '8', '--alpha', '16', '--lr', '1e-3', '--log-every', '96', '--seed', '4277009102', '--save', out / f'adapter-{family}.safetensors'], out / f'{phase}.log')
    elif phase in ('gen-base', 'gen-M', 'gen-G'):
        arm = phase[4:]
        rows = [json.loads(line) for line in pathlib.Path('w6-input/heldout.jsonl').read_text().splitlines()]
        if len(rows) != 80:
            raise ValueError('expected 80 prompts')
        with open(out / f'gen-{arm}.jsonl', 'w') as dest, open(out / f'duplicates-{arm}.jsonl', 'w') as duplicates:
            for row in rows:
                cmd = [bin_dir / 'generate_lora', '--model-dir', model, '--prompt', row['prompt'], '--max-tokens', '64', '--temperature', '0', '--seed', '1']
                if arm != 'base':
                    family = 'memory' if arm == 'M' else 'gtd'
                    cmd += ['--lora', out / f'adapter-{family}.safetensors']
                for repeat in range(2 if row['idx'] in (0, 1, 40, 41) else 1):
                    log = out / f'raw-{arm}-{row["idx"]}-{repeat}.log'
                    run(cmd, log)
                    text = log.read_bytes().decode('utf-8')
                    if text.count('--- Output ---\n') != 1 or text.count('\n--- Stats ---') != 1:
                        raise ValueError(f'missing/ambiguous output markers: {log}')
                    output, stats = text.split('--- Output ---\n', 1)[1].split('\n--- Stats ---', 1)
                    if (arm != 'base') != ('LoRA: ACTIVE' in stats.splitlines()):
                        raise ValueError(f'adapter activation marker differs from arm: {log}')
                    result = {k: row[k] for k in ('idx', 'family', 'prompt')}
                    result.update(output=output, stats=stats)
                    if repeat:
                        if output != original:
                            raise ValueError(f'K5: nondeterministic output {arm} idx={row["idx"]}')
                        duplicates.write(json.dumps(result) + '\n')
                        duplicates.flush()
                    else:
                        original = output
                        dest.write(json.dumps(result) + '\n')
                        dest.flush()
    elif phase == 'route':
        run([bin_dir / 'examples/prompt_router_vectors', 'w6-input/router.jsonl', out / 'vectors.json'], out / 'vectors.log')
        for arm in ('real', 'shuffled', 'cost', 'decisions'):
            run([bin_dir / 'examples/prompt_router', out / 'vectors.json', arm], out / f'route-{arm}.log')
    else:
        raise ValueError(f'unknown phase {phase}')
    receipt['ok'] = True
except BaseException as error:
    receipt['ok'] = False
    receipt['error'] = str(error)
    raise
finally:
    receipt['elapsed_seconds'] = time.monotonic() - start
    receipt['end'] = conditions()
    (out / f'conditions-{phase}.json').write_text(json.dumps(receipt, indent=2) + '\n')
PY
