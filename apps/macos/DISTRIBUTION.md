# Lattice Studio — Distribution Guide

## Building a distributable .app

Run the packaging script from the repo root:

```bash
cd lattice/
./apps/macos/scripts/package-app.sh
```

This will:
1. Run `swift build -c release` to produce the Swift instrument panel.
2. Build all 10 Rust engine binaries with `cargo build --release`: `quantize_q4`,
   `quantize_quarot`, `lattice`, `qwen35_generate`, `train_grad_full`, `generate_lora`,
   `eval_perplexity`, `embed`, `chat_metal`, and `lattice_serve`.
3. Assemble `apps/macos/dist/Lattice.app` with the engine binaries inside
   `Contents/Resources/bin/` so the app is fully self-contained.
4. Write `Contents/Info.plist` with bundle ID `ai.khive.lattice.studio`.
5. Ad-hoc codesign the bundle.
6. Produce `apps/macos/dist/Lattice.dmg` and `Lattice.zip`.

Options:

```bash
# Skip Swift rebuild (use existing .build/release/Lattice)
./apps/macos/scripts/package-app.sh --skip-build

# Skip Cargo rebuild (use existing target/release/ binaries)
./apps/macos/scripts/package-app.sh --skip-cargo

# Custom output directory
./apps/macos/scripts/package-app.sh --out /path/to/out
```

## Installing

Drag `Lattice.app` from the DMG to `/Applications`.

Requires macOS 14.0 (Sonoma) or later. The app is self-contained — no source
checkout, no Cargo, no Rust toolchain needed on the recipient machine.

## Gatekeeper caveat (ad-hoc signing)

The app is ad-hoc signed, which means macOS Gatekeeper will quarantine it on
first launch because it lacks a Developer ID certificate. Recipients must:

**Option 1 — right-click → Open:**
1. Right-click (or Ctrl-click) `Lattice.app` in Finder.
2. Choose "Open" from the context menu.
3. Click "Open" in the dialog. macOS remembers the exception.

**Option 2 — remove the quarantine xattr:**
```bash
xattr -dr com.apple.quarantine /Applications/Lattice.app
```

This is a one-time step. The app will open normally on subsequent launches.

## Upgrading to Developer ID signing (for App Store or notarized distribution)

When a $99/year Apple Developer Program membership is available, replace the
ad-hoc step in `package-app.sh` with a full Developer ID workflow:

```bash
# 1. Replace the ad-hoc sign step with a Developer ID certificate
DEVELOPER_ID="Developer ID Application: Haiyang Li (TEAMID)"
codesign --force --deep --options runtime \
         --entitlements apps/macos/LatticeStudio.entitlements \
         --sign "$DEVELOPER_ID" \
         "$BUNDLE"

# 2. Create a notarization credentials profile (one-time)
xcrun notarytool store-credentials "lattice-notary" \
    --apple-id "lhydyxzh@gmail.com" \
    --team-id "TEAMID" \
    --password "@keychain:AC_PASSWORD"

# 3. Submit for notarization and wait
xcrun notarytool submit "$DMG" \
    --keychain-profile "lattice-notary" \
    --wait

# 4. Staple the notarization ticket to the DMG
xcrun stapler staple "$DMG"
```

After notarization, Gatekeeper accepts the app on any Mac without the
right-click-Open workaround. Distribution via direct download or any store
works without restrictions.

An entitlements file (`LatticeStudio.entitlements`) must grant at minimum:
```xml
<key>com.apple.security.cs.allow-jit</key><false/>
<key>com.apple.security.cs.allow-unsigned-executable-memory</key><false/>
```

The engine binaries (Rust) do not use JIT, so hardened runtime is compatible.

## Regenerating the app icon

```bash
uv run apps/macos/scripts/generate-icon.py
```

Outputs `apps/macos/Resources/LatticeStudio.icns`. The packaging script picks
it up automatically, or re-runs the generator if the file is missing.

## Binary resolution order (inside the .app)

The Swift bridge resolves engine binaries in this order:
1. `Contents/Resources/bin/<name>` — bundled binary (used from /Applications).
2. `LATTICE_BIN_DIR` env var — dev override.
3. `<repo root>/target/release/<name>` — running from source checkout.
4. `cargo run --release` — compile-on-demand fallback.

This means the same app binary works both installed from the DMG and invoked
during development inside the source tree.
