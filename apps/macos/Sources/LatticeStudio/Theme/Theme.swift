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
// Governing laws (DESIGN.md):
//   1. Numbers never touch glass — every numeral sits on an opaque surface.
//   2. One accent (Signal Teal), spent only on movement / the single CTA.
//   3. Bold is spent on the NUMBERS (tabular mono), not on chrome.
enum Theme {

    // MARK: Palette — Dark is the home key; light is a true peer.
    enum Palette {
        static let canvas    = Color.adaptive(light: 0xF7F8FA, dark: 0x0A0B0D) // instrument face (opaque)
        static let panel     = Color.adaptive(light: 0xFFFFFF, dark: 0x121419) // raised center panel / well body
        static let wellSink  = Color.adaptive(light: 0xECEEF2, dark: 0x070809) // recessed fill inside a readout well
        static let hairline  = Color.adaptive(light: 0xDCDFE5, dark: 0x23262E) // 1px rules — depth via line
        static let ink       = Color.adaptive(light: 0x14161A, dark: 0xE8EAED) // primary text + numerals
        static let inkDim    = Color.adaptive(light: 0x5C636E, dark: 0x7C828D) // labels, units, axis ticks

        /// The one accent. Live trace, token stream, now-cursor, focus ring, the single CTA per screen.
        static let signal    = Color.adaptive(light: 0x00A892, dark: 0x00E5C7)
        static let signalGlow = Color.adaptive(light: 0x00A892, dark: 0x00E5C7, alpha: 0.12)

        static let amber     = Color.adaptive(light: 0xB8730A, dark: 0xFFB020) // regression / warning
        static let crimson   = Color.adaptive(light: 0xD3344A, dark: 0xFF4D5E) // hard failure only
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

        // Modular scale (pt): 11 · 13 · 15 · 21 · 34 · 56
        static let hero      = mono(56, .bold)   // the loss / compression / PPL headline
        static let heroAlt   = mono(34, .bold)
        static let heroMinor = mono(21, .semibold)
        static let wellValue = mono(15, .medium) // readout-well value
        static let readout   = mono(13)          // dense readouts
        static let cell      = mono(11)          // table cells
        static let title     = display(17, .semibold)
        static let body      = display(13, .regular)
        /// 11pt all-caps instrument label (LOSS, TOK/S, GRAD-NORM). Apply `.textCase(.uppercase)` + tracking.
        static let label     = display(11, .medium)
    }

    // MARK: Spacing — 8pt base grid; 4pt dense variant for table rows.
    enum Space {
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
    }

    // MARK: Corner radii — panels are RULED not carded (0px); wells/controls 6px; command bar 10px.
    enum Radius {
        static let panel: CGFloat = 0
        static let well: CGFloat = 6
        static let control: CGFloat = 6
        static let pill: CGFloat = 6
        static let commandBar: CGFloat = 10
    }

    // MARK: Motion — mechanical, never bouncy. Durations 120/180/240ms ease-out.
    enum Motion {
        static let tick: Double = 0.12
        static let focus: Double = 0.18
        static let pane: Double = 0.24
        /// Spring is reserved for EXACTLY one element: the adapter hot-swap fader.
        static let faderSpring = Animation.spring(response: 0.32, dampingFraction: 0.85)
        static let chartCommitHz: Double = 20
        static let numeralTickHz: Double = 8
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
