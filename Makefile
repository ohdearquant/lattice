.PHONY: check clippy test fmt fmt-check build clean ci publish publish-dry lint-docs bench-ci bench-gate bench-compare bench-decode-slopefit

check:
	cargo check --workspace

clippy:
	cargo clippy --workspace -- -D warnings

test:
	cargo test --workspace

fmt:
	cargo fmt --all
	deno fmt **/*.md

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

# ADR-058: run the same CPU benches CI does, save as new baseline.
bench-ci:
	cargo bench -p lattice-inference --bench elementwise_cpu_bench -- --save-baseline local --noplot
	cargo bench -p lattice-embed --bench simd -- --save-baseline local --noplot

# ADR-058: compare current CPU bench results against the perf-baselines branch
# checked into ./.cache/perf-baselines. Auto-detects local arch label.
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
		mkdir -p target/criterion; \
		cp -r .cache/perf-baselines/$$arch/. target/criterion/ 2>/dev/null || { echo "no baseline for $$arch"; exit 1; }; \
		cargo bench -p lattice-inference --bench elementwise_cpu_bench -- --baseline base --noplot; \
		cargo bench -p lattice-embed --bench simd -- --baseline base --noplot; \
		python3 scripts/perf-bench-gate.py target/criterion "$$arch-local"

# E2E parity: HF transformers (reference) vs lattice (greedy token agreement).
# Requires: pip install torch transformers tokenizers
e2e-parity:
	cargo build --release --bin qwen35_generate -p lattice-inference --features f16
	python3 scripts/e2e_parity_check.py

# ADR-064: measure decode slope/intercept and output JSON.
# Usage: make bench-decode                   (default 5 runs, ctx 64-1024)
#        RUNS=3 make bench-decode            (faster)
bench-decode:
	./scripts/bench_decode_slope.sh

# ADR-064 Phase-0: decode slope/intercept fit (Theil-Sen + bootstrap CI).
# Usage: make bench-decode-slopefit                     (smoke grid {64,256,512})
#        SLOPEFIT_FULL=1 make bench-decode-slopefit     (full grid up to 16K ctx)
#        SLOPEFIT_CONTEXTS="64 512 2048" make bench-decode-slopefit
bench-decode-slopefit:
	./scripts/bench_decode_slopefit.sh

# A/B benchmark across git refs. Uses worktree for base, leaves working tree untouched.
# Usage: make bench-compare                     (origin/main vs HEAD)
#        make bench-compare BASE=main HEAD=pr/x (explicit refs)
bench-compare:
	./scripts/bench-compare.sh $(or $(BASE),origin/main) $(or $(HEAD),HEAD)
