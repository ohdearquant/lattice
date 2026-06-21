#!/usr/bin/env bash
# package-app.sh — Build and package LatticeStudio.app + .dmg + .zip
#
# Usage:
#   ./scripts/package-app.sh [--out <dir>] [--skip-build] [--skip-cargo]
#
# Options:
#   --out <dir>       Output directory (default: apps/macos/dist/)
#   --skip-build      Skip `swift build -c release` (use existing .build/release/LatticeStudio)
#   --skip-cargo      Skip cargo builds (use existing target/release/ binaries)
#
# Idempotent: re-running overwrites dist/ cleanly.
#
# Signing: ad-hoc codesign only (no Developer ID required).
# Recipients must right-click → Open, or: xattr -dr com.apple.quarantine LatticeStudio.app
# See DISTRIBUTION.md for the Developer ID upgrade path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"       # apps/macos/
REPO_ROOT="$(cd "$MACOS_DIR/../.." && pwd)"     # lattice/
OUT_DIR="$MACOS_DIR/dist"
SKIP_BUILD=false
SKIP_CARGO=false

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)   OUT_DIR="$(realpath "$2")"; shift 2 ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --skip-cargo)  SKIP_CARGO=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

APP_NAME="LatticeStudio"
BUNDLE="$OUT_DIR/$APP_NAME.app"
VERSION="0.3.0"
BUNDLE_ID="ai.khive.lattice.studio"

echo "==> Package $APP_NAME v$VERSION"
echo "    Repo:   $REPO_ROOT"
echo "    Output: $OUT_DIR"

# --- Step 1: Swift release build ---
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "==> [1/6] swift build -c release"
    swift build -c release --package-path "$MACOS_DIR"
fi
SWIFT_BIN="$MACOS_DIR/.build/release/$APP_NAME"
if [ ! -x "$SWIFT_BIN" ]; then
    echo "ERROR: $SWIFT_BIN not found or not executable"
    exit 1
fi
echo "    Swift binary: $(du -sh "$SWIFT_BIN" | awk '{print $1}')"

# --- Step 2: Cargo release builds ---
if [ "$SKIP_CARGO" = false ]; then
    echo ""
    echo "==> [2/6] cargo build --release (8 engine binaries)"
    # lattice-inference binaries (no extra features)
    for BIN in quantize_q4 quantize_quarot lattice qwen35_generate; do
        echo "    cargo build --release -p lattice-inference --bin $BIN"
        cargo build --release -p lattice-inference --bin "$BIN" \
            --manifest-path "$REPO_ROOT/Cargo.toml"
    done
    # lattice-tune: train_grad_full needs train-backward feature
    echo "    cargo build --release -p lattice-tune --bin train_grad_full --features train-backward"
    cargo build --release -p lattice-tune --bin train_grad_full \
        --features train-backward \
        --manifest-path "$REPO_ROOT/Cargo.toml"
    # lattice-tune: generate_lora needs safetensors,inference-hook
    echo "    cargo build --release -p lattice-tune --bin generate_lora --features safetensors,inference-hook"
    cargo build --release -p lattice-tune --bin generate_lora \
        --features "safetensors,inference-hook" \
        --manifest-path "$REPO_ROOT/Cargo.toml"
    # lattice-inference: eval_perplexity needs f16,metal-gpu (CPU bf16 + Metal Q4/QuaRot PPL)
    echo "    cargo build --release -p lattice-inference --bin eval_perplexity --features f16,metal-gpu"
    cargo build --release -p lattice-inference --bin eval_perplexity \
        --features "f16,metal-gpu" \
        --manifest-path "$REPO_ROOT/Cargo.toml"
    # lattice-embed: embed CLI (native default features)
    echo "    cargo build --release -p lattice-embed --bin embed"
    cargo build --release -p lattice-embed --bin embed \
        --manifest-path "$REPO_ROOT/Cargo.toml"
fi

TARGET_RELEASE="$REPO_ROOT/target/release"
for BIN in quantize_q4 quantize_quarot lattice qwen35_generate train_grad_full generate_lora eval_perplexity embed; do
    if [ ! -x "$TARGET_RELEASE/$BIN" ]; then
        echo "ERROR: $TARGET_RELEASE/$BIN not found"
        exit 1
    fi
done
echo "    All 8 engine binaries present"

# --- Step 3: Construct .app bundle ---
echo ""
echo "==> [3/6] Construct $APP_NAME.app"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS"
mkdir -p "$BUNDLE/Contents/Resources/bin"

# Executable
cp "$SWIFT_BIN" "$BUNDLE/Contents/MacOS/$APP_NAME"
chmod +x "$BUNDLE/Contents/MacOS/$APP_NAME"

# Engine binaries
for BIN in quantize_q4 quantize_quarot lattice qwen35_generate train_grad_full generate_lora eval_perplexity embed; do
    cp "$TARGET_RELEASE/$BIN" "$BUNDLE/Contents/Resources/bin/$BIN"
    chmod +x "$BUNDLE/Contents/Resources/bin/$BIN"
done
echo "    Copied 8 engine binaries → Contents/Resources/bin/"

# App icon
ICNS_SRC="$MACOS_DIR/Resources/LatticeStudio.icns"
if [ -f "$ICNS_SRC" ]; then
    cp "$ICNS_SRC" "$BUNDLE/Contents/Resources/LatticeStudio.icns"
    echo "    Copied icon"
else
    echo "    WARNING: $ICNS_SRC not found, regenerating..."
    uv run "$MACOS_DIR/scripts/generate-icon.py" --out "$ICNS_SRC"
    cp "$ICNS_SRC" "$BUNDLE/Contents/Resources/LatticeStudio.icns"
fi

# --- Step 4: Info.plist ---
echo ""
echo "==> [4/6] Write Info.plist"
cat > "$BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>Lattice Studio</string>
    <key>CFBundleDisplayName</key>
    <string>Lattice Studio</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>LatticeStudio</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2026 khive AI. All rights reserved.</string>
</dict>
</plist>
PLIST
echo "    Info.plist written (bundle id: $BUNDLE_ID)"

# --- Step 5: Ad-hoc codesign ---
echo ""
echo "==> [5/6] Ad-hoc codesign"
codesign --force --deep --options runtime --sign - "$BUNDLE"
codesign --verify --deep --strict "$BUNDLE"
echo "    Codesign verified"

# --- Step 6: DMG + ZIP ---
echo ""
echo "==> [6/6] Create distributable artifacts"
DMG="$OUT_DIR/$APP_NAME.dmg"
ZIP="$OUT_DIR/$APP_NAME.zip"

# DMG
hdiutil create \
    -volname "Lattice Studio" \
    -srcfolder "$BUNDLE" \
    -ov \
    -format UDZO \
    "$DMG" 2>&1 | grep -v "^hdiutil:"
echo "    DMG: $(du -sh "$DMG" | awk '{print $1}')  $DMG"

# ZIP fallback
(cd "$OUT_DIR" && zip -qr "$APP_NAME.zip" "$APP_NAME.app")
echo "    ZIP: $(du -sh "$ZIP" | awk '{print $1}')  $ZIP"

echo ""
echo "==> Done. Artifacts in $OUT_DIR/"
du -sh "$BUNDLE" "$DMG" "$ZIP"
echo ""
echo "Distribution note: Recipients must right-click → Open on first launch (Gatekeeper)."
echo "Or: xattr -dr com.apple.quarantine '$BUNDLE'"
