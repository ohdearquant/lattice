.PHONY: check clippy test test-timing fmt fmt-check build clean ci publish publish-dry publish-npm publish-npm-dry lint-docs bench-ci bench-gate bench-compare bench-agentic bench-agentic-quick wasm-parity e2e-parity bench-decode-slopefit

check:
	cargo check --workspace

clippy:
	cargo clippy --workspace -- -D warnings

test:
	cargo test --workspace

# The suites carrying wall-clock regression checks (cached-SIMD-config lookup,
# 1000-batch dot product, FANN inference). Those assertions are skipped when the
# CI env var is set, because absolute-time bounds flake under shared-CI
# scheduling contention. Run this on an idle host to exercise them for real.
test-timing:
	cargo test -p lattice-embed --test integration
	cargo test -p lattice-fann --lib

fmt:
	cargo fmt --all
	./scripts/lint-docs.sh --format

fmt-check:
	cargo fmt --all -- --check

build:
	cargo build --workspace --release

clean:
	cargo clean

lint-docs:
	./scripts/lint-docs.sh

ci:
	./scripts/ci.sh

publish-dry:
	./scripts/publish.sh --dry-run

publish:
	./scripts/publish.sh

# npm embedding packages (@khive-ai/lattice-embed + @khive-ai/lattice-embed-wasm).
publish-npm-dry:
	./scripts/publish-npm.sh --dry-run

publish-npm:
	./scripts/publish-npm.sh

# ADR-058: run the same CPU benches CI does, save as new baseline.
bench-ci:
	cargo bench -p lattice-inference --bench elementwise_cpu_bench -- --save-baseline local --noplot
	cargo bench -p lattice-embed --bench simd -- --save-baseline local --noplot

# ADR-058: compare current CPU bench results against the perf-baselines branch
# checked into ./.cache/perf-baselines. Its rc enforces measurement completeness
# and regressions, but without run provenance the report is unsuitable as
# benchmark evidence. Auto-detects local arch label.
bench-gate:
	@if [ ! -d .cache/perf-baselines ]; then \
		git clone --depth=1 --branch=perf-baselines \
			$$(git remote get-url origin) .cache/perf-baselines || \
			{ echo "no perf-baselines branch yet — run bench-update.yml on main first"; exit 1; }; \
	else \
		git -C .cache/perf-baselines pull --ff-only; \
	fi
	@arch=$$(uname -m | sed 's/arm64/aarch64/')-$$(uname -s | tr A-Z a-z); \
		echo "arch: $$arch"; \
		root=.cache/bench-gate-criterion; \
		inference_root=$$root/inference/criterion; \
		embed_root=$$root/embed/criterion; \
		rm -rf "$$root"; \
		mkdir -p "$$inference_root" "$$embed_root"; \
		baseline=.cache/perf-baselines/$$arch; \
		[ -d "$$baseline" ] || { echo "no baseline for $$arch"; exit 1; }; \
		cargo bench -p lattice-inference --bench elementwise_cpu_bench -- --list \
			> "$$root/inference-bench-list"; \
		sed -n 's/: benchmark$$//p' "$$root/inference-bench-list" \
			> "$$root/inference-bench-ids"; \
		while IFS= read -r bench; do \
			[ -d "$$baseline/$$bench" ] || continue; \
			mkdir -p "$$inference_root/$$(dirname "$$bench")"; \
			cp -R "$$baseline/$$bench" "$$inference_root/$$bench"; \
		done < "$$root/inference-bench-ids"; \
		cargo bench -p lattice-embed --bench simd -- --list \
			> "$$root/embed-bench-list"; \
		sed -n 's/: benchmark$$//p' "$$root/embed-bench-list" \
			> "$$root/embed-bench-ids"; \
		while IFS= read -r bench; do \
			[ -d "$$baseline/$$bench" ] || continue; \
			mkdir -p "$$embed_root/$$(dirname "$$bench")"; \
			cp -R "$$baseline/$$bench" "$$embed_root/$$bench"; \
		done < "$$root/embed-bench-ids"; \
		CRITERION_HOME="$$inference_root" cargo bench -p lattice-inference --bench elementwise_cpu_bench -- --baseline base --noplot; \
		CRITERION_HOME="$$embed_root" cargo bench -p lattice-embed --bench simd -- --baseline base --noplot; \
		echo "UNSUITABLE AS BENCHMARK EVIDENCE: local bench-gate has no run provenance"; \
		rc=0; \
		python3 scripts/perf-bench-gate.py "$$inference_root" "$$arch-local/lattice-inference:elementwise_cpu_bench" \
			--target lattice-inference:elementwise_cpu_bench --require-measurements || rc=$$?; \
		python3 scripts/perf-bench-gate.py "$$embed_root" "$$arch-local/lattice-embed:simd" \
			--target lattice-embed:simd --require-measurements || gate_rc=$$?; \
		if [ "$${gate_rc:-0}" -eq 2 ] || [ "$$rc" -eq 2 ]; then exit 2; fi; \
		if [ "$${gate_rc:-0}" -ne 0 ] || [ "$$rc" -ne 0 ]; then exit 1; fi

# E2E parity: HF transformers (reference) vs lattice (greedy token agreement).
# Requires: pip install torch transformers tokenizers
e2e-parity:
	cargo build --release --bin qwen35_generate -p lattice-inference --features f16
	python3 scripts/e2e_parity_check.py

# ADR-064 Phase-0: decode slope/intercept fit (Theil-Sen + bootstrap CI).
# Usage: make bench-decode-slopefit                     (smoke grid {64,256,512})
#        SLOPEFIT_FULL=1 make bench-decode-slopefit     (full grid up to 16K ctx)
#        SLOPEFIT_CONTEXTS="64 512 2048" make bench-decode-slopefit
bench-decode-slopefit:
	./scripts/bench_decode_slopefit.sh

# A/B benchmark across git refs. Uses worktree for base, leaves working tree untouched.
# Usage: make bench-compare                                           (origin/main vs HEAD)
#        make bench-compare BASE=main HEAD=pr/x                       (explicit refs)
#        make bench-compare BENCH_GROUPS_INFERENCE='rms_norm|gelu'    (Criterion filter)
#        make bench-compare BENCH_GROUPS_EMBED='simd_dot_product'     (Criterion filter)
bench-compare:
	BENCH_GROUPS_INFERENCE="$(value BENCH_GROUPS_INFERENCE)" BENCH_GROUPS_EMBED="$(value BENCH_GROUPS_EMBED)" ./scripts/bench-compare.sh $(or $(BASE),origin/main) $(or $(HEAD),HEAD)

# Agentic-workload benchmark: lattice vs ollama vs MLX at 1000/2000/4000-token context.
# Prereqs: bench_decode_ab binary built, ollama serve running, mlx_lm available.
# Build binary: cargo build --release --bin bench_decode_ab -p lattice-inference --features "f16,metal-gpu"
bench-agentic:
	uv run --with mlx-lm python3 scripts/bench_decode_harness.py run --profile agentic --sweep --allow-missing-engine

# Fast sanity check: ctx=1000 only, 3 runs.
bench-agentic-quick:
	uv run --with mlx-lm python3 scripts/bench_decode_harness.py run --profile agentic --ctx 1000 --runs 3 --allow-missing-engine

# wasm embedding parity gate: builds lattice-embed for wasm32, runs it against
# the same HF-reference goldens + native-lattice reference as the embed
# parity test. Skip-graceful if node/wasm-bindgen/weights are absent; set
# LATTICE_WASM_PARITY_ENFORCE=1 to fail closed instead (CI does this).
wasm-parity:
	./scripts/wasm-parity.sh
