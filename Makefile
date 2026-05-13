.PHONY: check clippy test fmt fmt-check build clean ci publish publish-dry lint-docs

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
