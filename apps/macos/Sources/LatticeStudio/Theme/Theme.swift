import SwiftUI
import AppKit

// MARK: - Color helpers (code-defined adaptive palette — no asset catalog, so `swift build` is hermetic)

extension NSColor {
    convenience init(hex: UInt32, alpha: CGFloat = 1.0) {
        let r = CGFloat((hex >> 16) & 0xFF) / 255.0
        let g = CGFloat((hex >> 8) & 0xFF) / 255.0
        let b = CGFloat(hex & 0xFF) / 255.0
        self.init(srgbRed: r, green: g, blue: b, alpha: alpha)
    }
}

extension Color {
    /// Resolves light/dark at the AppKit layer so the value tracks system appearance automatically.
    static func adaptive(light: UInt32, dark: UInt32, alpha: CGFloat = 1.0) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            return NSColor(hex: isDark ? dark : light, alpha: alpha)
        })
    }
}

// MARK: - Theme: the single source of truth for the Lattice Instrument visual language.
//
// Graphite Signal Lab — dark graphite surfaces, 10pt cards, white-opacity hairlines.
// One cyan accent (signal/CTA). Monospaced numerals on opaque surfaces.
//
// Governing laws (DESIGN.md):
//   1. Numbers never touch glass — every numeral sits on an opaque surface.
//   2. One accent (Signal Teal / #48D8C4), spent only on movement / the single CTA.
//   3. Bold is spent on the NUMBERS (tabular mono), not on chrome.
enum Theme {

    // MARK: Palette — Dark is the home key; light is a true peer.
    enum Palette {
        // ── Base surfaces ──────────────────────────────────────────────────────────

        /// Window background beneath all content.
        static let window          = Color.adaptive(light: 0xF4F6F8, dark: 0x090C10)

        /// Main screen canvas.
        static let canvas          = Color.adaptive(light: 0xEEF1F4, dark: 0x0D1117)

        /// Solid sidebar fallback when Reduce Transparency is enabled.
        static let sidebarFallback = Color.adaptive(light: 0xF0F3F7, dark: 0x10151C)

        /// Standard panels and cards (spec: surface). Replaces old `panel`.
        static let panel           = Color.adaptive(light: 0xFFFFFF, dark: 0x121820)

        /// Controls, selected cards, elevated sections.
        static let surfaceRaised   = Color.adaptive(light: 0xF7F9FB, dark: 0x171E27)

        /// Hovered rows and controls.
        static let surfaceHover    = Color.adaptive(light: 0xEDF2F5, dark: 0x1C2530)

        /// Charts, logs, code and recessed readouts (spec: surfaceInset). Replaces old `wellSink`.
        static let wellSink        = Color.adaptive(light: 0xE8EDF2, dark: 0x080B0F)

        /// Alias kept for code that imports surfaceInset directly.
        static var surfaceInset: Color { wellSink }

        /// Disabled control fill.
        static let surfaceDisabled = Color.adaptive(light: 0xEBEFF3, dark: 0x11161C)

        // ── Text ───────────────────────────────────────────────────────────────────

        /// Primary labels and prose. Replaces old `ink`.
        static let ink             = Color.adaptive(light: 0x11151A, dark: 0xF2F5F7)

        /// Alias kept for direct textPrimary references.
        static var textPrimary: Color { ink }

        /// Supporting text. Replaces old `inkDim`.
        static let inkDim          = Color.adaptive(light: 0x56616D, dark: 0xA7B0BA)

        /// Alias kept for direct textSecondary references.
        static var textSecondary: Color { inkDim }

        /// Metadata, inactive labels, axis values.
        /// WCAG AA fix: 0x646E79 = 4.58:1 on canvas 0xEEF1F4; 0x79838F = 4.92:1 on canvas 0x0D1117.
        static let textTertiary    = Color.adaptive(light: 0x646E79, dark: 0x79838F)

        /// Disabled controls.
        static let textDisabled    = Color.adaptive(light: 0x9AA3AD, dark: 0x4E5965)

        // ── Accent ─────────────────────────────────────────────────────────────────

        /// The one accent. Live trace, token stream, now-cursor, focus ring, the single CTA per screen.
        static let signal          = Color.adaptive(light: 0x078F82, dark: 0x48D8C4)

        /// Alias kept for direct accent references.
        static var accent: Color { signal }

        /// Hovered primary action.
        static let accentHover     = Color.adaptive(light: 0x0AA493, dark: 0x63E2D1)

        /// Pressed primary action.
        static let accentActive    = Color.adaptive(light: 0x08786F, dark: 0x2EB9A7)

        /// Text/icons on accent fill.
        /// WCAG AA fix (light): 0x0A0D11 = 4.88:1 on signal 0x078F82 (white was only 3.99:1).
        /// Dark stays 0x04110F = 10.89:1 on signal 0x48D8C4.
        static let onAccent        = Color.adaptive(light: 0x0A0D11, dark: 0x04110F)

        /// Accent glow fill — accent at 12% opacity.
        static let signalGlow      = Color.adaptive(light: 0x078F82, dark: 0x48D8C4, alpha: 0.12)

        // ── State semantics ────────────────────────────────────────────────────────

        /// Idle engine state.
        static let idle            = Color.adaptive(light: 0x7D8793, dark: 0x7D8793)

        /// Active run.
        static let running         = Color.adaptive(light: 0x078F82, dark: 0x48D8C4)

        /// Completed, valid, improvement.
        static let success         = Color.adaptive(light: 0x3BAF5C, dark: 0x6BD58C)

        /// Validation warning, partial issue.
        static let amber           = Color.adaptive(light: 0xB8730A, dark: 0xF2B85B)

        /// Alias kept for direct warning references.
        static var warning: Color { amber }

        /// Failed, invalid, regression.
        static let crimson         = Color.adaptive(light: 0xC42032, dark: 0xFF6B73)

        /// Alias kept for direct error references.
        static var error: Color { crimson }

        // ── Neutral control ────────────────────────────────────────────────────────

        /// Neutral steel for interactive controls (slider fill, segmented selection, toggle-on,
        /// field caret). Spent so the accent above stays reserved for live data + the single CTA.
        static let control         = Color.adaptive(light: 0x8A909B, dark: 0x6B7280)

        // ── Hairline ───────────────────────────────────────────────────────────────

        /// 1px ruled borders — depth via line, not shadow.
        static let hairline        = Color.adaptive(light: 0xDCDFE5, dark: 0x23262E)

        // ── Opacity-based border / overlay helpers (white-relative, use on dark surfaces) ──

        /// Default card border — white at 8%.
        static let borderStandard  = Color.white.opacity(0.08)

        /// Selected/strong card border — white at 14%.
        static let borderStrong    = Color.white.opacity(0.14)

        /// Hovered row/control overlay — white at 4%.
        static let hoverOverlay    = Color.white.opacity(0.04)

        /// Accent selection fill — accent at 12%.
        static let selectionFill   = Color.adaptive(light: 0x078F82, dark: 0x48D8C4, alpha: 0.12)

        /// Accent selection border — accent at 35%.
        static let selectionBorder = Color.adaptive(light: 0x078F82, dark: 0x48D8C4, alpha: 0.35)

        /// Accent focus ring — accent at 55%.
        static let focusRing       = Color.adaptive(light: 0x078F82, dark: 0x48D8C4, alpha: 0.55)

        /// Chart minor grid — white at 4%.
        static let chartGridMinor  = Color.white.opacity(0.04)

        /// Chart major grid — white at 8%.
        static let chartGridMajor  = Color.white.opacity(0.08)
    }

    // MARK: Typography — mono is the signature. SF Mono today; swap `mono()` to bundled JetBrains Mono later.
    enum Fonts {
        /// Tabular monospace for every numeral / data cell. `.monospacedDigit()` is applied at use sites.
        static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
            .system(size: size, weight: weight, design: .monospaced)
        }
        /// SF Pro for titles, labels, prose.
        static func display(_ size: CGFloat, _ weight: Font.Weight = .semibold) -> Font {
            .system(size: size, weight: weight, design: .default)
        }

        // ── Legacy modular scale (preserved for backward compat) ──────────────────
        // Modular scale (pt): 11 · 13 · 15 · 21 · 34 · 56
        static let hero      = mono(56, .bold)    // the loss / compression / PPL headline
        static let heroAlt   = mono(34, .bold)
        static let heroMinor = mono(21, .semibold)
        static let wellValue = mono(15, .medium)  // readout-well value
        static let readout   = mono(13)           // dense readouts
        static let cell      = mono(11)           // table cells
        static let title     = display(17, .semibold)
        static let body      = display(13, .regular)
        /// 11pt all-caps instrument label (LOSS, TOK/S, GRAD-NORM). Apply `.textCase(.uppercase)` + tracking.
        static let label     = display(11, .medium)

        // ── Graphite Signal Lab role scale (new additions) ────────────────────────

        /// Screen/section title — Models, Train, Quantize.
        static let screenTitle     = display(22, .semibold)

        /// Selected model or run header inside an inspector.
        static let inspectorTitle  = display(17, .semibold)

        /// Group labels — always rendered uppercase + 0.85 tracking at use site.
        static let sectionLabel    = display(11, .semibold)

        /// Row names and values (medium weight body).
        static let bodyStrong      = display(13, .medium)

        /// Buttons, pickers — 12pt medium.
        /// (Named `controlText` to avoid shadowing the `control` palette token.)
        static let controlText     = display(12, .medium)

        /// Supporting text — 11pt regular.
        static let caption         = display(11, .regular)

        /// Metric labels, metadata — 10pt medium (+0.45 tracking at use site).
        static let micro           = display(10, .medium)

        /// Large loss / error / size headline — 20pt SF Mono medium.
        static let largeMetric     = mono(20, .medium)

        /// Counts, dimensions — 13pt SF Mono medium.
        static let metric          = mono(13, .medium)

        /// Params, bytes, duration table cells — 12pt SF Mono regular.
        static let tableNumeric    = mono(12)

        /// Commands and JSON — 12pt SF Mono regular.
        static let codeFont        = mono(12)

        /// Streaming console — 11pt SF Mono regular.
        static let logFont         = mono(11)
    }

    // MARK: Spacing — 4pt base grid (Graphite Signal Lab); legacy 8pt names preserved.
    enum Space {
        // ── Legacy 8pt-grid names (kept for backward compat) ─────────────────────
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16   // internal panel padding
        static let xl: CGFloat = 24   // section gutter
        static let xxl: CGFloat = 32

        static let railWidth: CGFloat = 220
        static let inspectorWidth: CGFloat = 300
        static let rowHeight: CGFloat = 28
        static let rowHeightComfortable: CGFloat = 32
        static let labelTracking: CGFloat = 0.6

        // ── 4pt scale additions ───────────────────────────────────────────────────
        static let space1: CGFloat  =  4
        static let space2: CGFloat  =  8
        static let space3: CGFloat  = 12
        static let space4: CGFloat  = 16
        static let space5: CGFloat  = 20
        static let space6: CGFloat  = 24
        static let space8: CGFloat  = 32
        static let space10: CGFloat = 40
        static let space12: CGFloat = 48

        // ── Layout constants ──────────────────────────────────────────────────────
        static let sidebarMin: CGFloat       = 196
        static let sidebarIdeal: CGFloat     = 212
        static let sidebarMax: CGFloat       = 240
        static let configRail: CGFloat       = 320
        static let detailInspector: CGFloat  = 300
        static let controlHeight: CGFloat    = 30
        static let controlHeightCompact: CGFloat = 26
        static let controlHeightLarge: CGFloat   = 34
        static let chatMaxWidth: CGFloat     = 920
        static let dataMaxWidth: CGFloat     = 1480
        static let emptyStateMaxWidth: CGFloat   = 360
    }

    // MARK: Corner radii — Graphite Signal Lab: panels are 10pt cards; wells 8pt.
    enum Radius {
        /// Standard card / panel / chart container (was 0 — now 10pt continuous cards).
        static let panel: CGFloat = 10

        /// Readout well / metric well (was 6 — now 8pt).
        static let well: CGFloat = 8

        /// Buttons, text fields, segmented items.
        static let control: CGFloat = 6

        /// Status capsules (backward-compat alias; use `.infinity` for a true capsule at use site).
        static let pill: CGFloat = 6

        /// Command / composer bar.
        static let commandBar: CGFloat = 10

        /// Compact badges and tiny indicators.
        static let badge: CGFloat = 4
    }

    // MARK: Motion — mechanical, never bouncy.
    enum Motion {
        static let tick: Double = 0.12
        static let focus: Double = 0.18
        static let pane: Double = 0.24
        /// Spring is reserved for EXACTLY one element: the adapter hot-swap fader.
        static let faderSpring = Animation.spring(response: 0.32, dampingFraction: 0.85)
        static let chartCommitHz: Double = 20
        static let numeralTickHz: Double = 8

        // ── Graphite Signal Lab timing additions ──────────────────────────────────
        /// Hover and selection — 100–120 ms ease-out.
        static let hover: Double = 0.11

        /// Button pressed state — 60–80 ms.
        static let press: Double = 0.07

        /// Metric numeric transition — 120–160 ms.
        static let metric: Double = 0.14

        /// Progress interpolation — 120–180 ms.
        static let progress: Double = 0.15
    }
}

// MARK: - Reusable view modifiers expressing the laws

extension View {
    /// An OPAQUE instrument label: 11pt all-caps, dimmed, tracked. For LOSS / TOK/S / GRAD-NORM.
    func instrumentLabel() -> some View {
        self.font(Theme.Fonts.label)
            .textCase(.uppercase)
            .tracking(Theme.Space.labelTracking)
            .foregroundStyle(Theme.Palette.inkDim)
    }
}
