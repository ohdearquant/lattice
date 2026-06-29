import SwiftUI

// MARK: - MarkdownText
//
// Lightweight, dependency-free Markdown renderer for chat responses. Models emit Markdown
// (headings, lists, fenced code, **bold**, `code`), which a plain `Text` shows as literal
// "###" / "```" markers and wraps oddly. This splits the source into blocks and renders each
// with the right layout, using `AttributedString(markdown:)` for inline styling.
//
// Not a full CommonMark implementation: it covers the syntax that actually shows up in model
// output and degrades to plain text for anything it doesn't recognize. Safe on partial input
// (streaming) — an unterminated code fence renders its body as code.

struct MarkdownText: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.lg) {
            ForEach(Array(MarkdownParser.parse(text).enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func view(for block: MarkdownBlock) -> some View {
        switch block {
        case let .heading(level, content):
            MarkdownParser.inline(content)
                .font(.system(size: headingSize(level), weight: .semibold))
                .foregroundStyle(Theme.Palette.ink)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

        case let .paragraph(content):
            MarkdownParser.inline(content)
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.ink)
                .lineSpacing(5)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

        case let .list(items, ordered):
            VStack(alignment: .leading, spacing: Theme.Space.xs) {
                ForEach(Array(items.enumerated()), id: \.offset) { idx, item in
                    HStack(alignment: .firstTextBaseline, spacing: Theme.Space.sm) {
                        Text(ordered ? "\(idx + 1)." : "•")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .monospacedDigit()
                        MarkdownParser.inline(item)
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.ink)
                            .lineSpacing(5)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

        case let .code(content):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(content)
                    .font(.system(size: 12, weight: .regular, design: .monospaced))
                    .foregroundStyle(Theme.Palette.ink)
                    .textSelection(.enabled)
                    .padding(Theme.Space.sm)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .fill(Theme.Palette.wellSink)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
            )

        case let .quote(content):
            HStack(alignment: .top, spacing: Theme.Space.sm) {
                RoundedRectangle(cornerRadius: 1)
                    .fill(Theme.Palette.hairline)
                    .frame(width: 2)
                MarkdownParser.inline(content)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: return 20
        case 2: return 17
        case 3: return 15
        default: return 14
        }
    }
}

// MARK: - Block model

enum MarkdownBlock {
    case heading(level: Int, text: String)
    case paragraph(String)
    case list(items: [String], ordered: Bool)
    case code(String)
    case quote(String)
}

// MARK: - Parser

enum MarkdownParser {
    /// Render inline Markdown (bold, italic, inline code, links) to a `Text`, preserving
    /// whitespace. Falls back to plain text if parsing fails (e.g. partial streaming input).
    static func inline(_ s: String) -> Text {
        let opts = AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: false,
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible
        )
        if let attr = try? AttributedString(markdown: s, options: opts) {
            return Text(attr)
        }
        return Text(s)
    }

    /// Split a Markdown document into renderable blocks. Line-based: handles fenced code,
    /// ATX headings, unordered/ordered lists, blockquotes, and blank-line-separated paragraphs.
    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        let lines = text.components(separatedBy: "\n")

        var paragraph: [String] = []
        func flushParagraph() {
            if !paragraph.isEmpty {
                blocks.append(.paragraph(paragraph.joined(separator: "\n")))
                paragraph.removeAll()
            }
        }

        var i = 0
        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Fenced code block: ``` ... ``` (language tag after the opening fence is dropped)
            if trimmed.hasPrefix("```") {
                flushParagraph()
                var code: [String] = []
                i += 1
                while i < lines.count && !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code.append(lines[i])
                    i += 1
                }
                i += 1 // consume the closing fence (or run off the end on partial input)
                blocks.append(.code(code.joined(separator: "\n")))
                continue
            }

            // ATX heading: #..###### followed by a space
            if let h = headingMatch(trimmed) {
                flushParagraph()
                blocks.append(.heading(level: h.level, text: h.text))
                i += 1
                continue
            }

            // Unordered list: collect consecutive -, *, + items
            if isUnorderedItem(trimmed) {
                flushParagraph()
                var items: [String] = []
                while i < lines.count, isUnorderedItem(lines[i].trimmingCharacters(in: .whitespaces)) {
                    items.append(stripUnordered(lines[i].trimmingCharacters(in: .whitespaces)))
                    i += 1
                }
                blocks.append(.list(items: items, ordered: false))
                continue
            }

            // Ordered list: collect consecutive "N." items
            if isOrderedItem(trimmed) {
                flushParagraph()
                var items: [String] = []
                while i < lines.count, isOrderedItem(lines[i].trimmingCharacters(in: .whitespaces)) {
                    items.append(stripOrdered(lines[i].trimmingCharacters(in: .whitespaces)))
                    i += 1
                }
                blocks.append(.list(items: items, ordered: true))
                continue
            }

            // Blockquote: collect consecutive "> " lines
            if trimmed.hasPrefix(">") {
                flushParagraph()
                var quote: [String] = []
                while i < lines.count, lines[i].trimmingCharacters(in: .whitespaces).hasPrefix(">") {
                    var q = lines[i].trimmingCharacters(in: .whitespaces)
                    q.removeFirst()
                    quote.append(q.trimmingCharacters(in: .whitespaces))
                    i += 1
                }
                blocks.append(.quote(quote.joined(separator: "\n")))
                continue
            }

            // Blank line ends a paragraph
            if trimmed.isEmpty {
                flushParagraph()
                i += 1
                continue
            }

            paragraph.append(line)
            i += 1
        }
        flushParagraph()
        return blocks
    }

    private static func headingMatch(_ s: String) -> (level: Int, text: String)? {
        guard s.hasPrefix("#") else { return nil }
        var level = 0
        for ch in s {
            if ch == "#" { level += 1 } else { break }
        }
        guard level >= 1, level <= 6 else { return nil }
        let rest = s.dropFirst(level)
        guard rest.hasPrefix(" ") else { return nil }
        return (level, rest.trimmingCharacters(in: .whitespaces))
    }

    private static func isUnorderedItem(_ s: String) -> Bool {
        s.hasPrefix("- ") || s.hasPrefix("* ") || s.hasPrefix("+ ")
    }

    private static func stripUnordered(_ s: String) -> String {
        String(s.dropFirst(2)).trimmingCharacters(in: .whitespaces)
    }

    private static func isOrderedItem(_ s: String) -> Bool {
        guard let dot = s.firstIndex(of: ".") else { return false }
        let num = s[s.startIndex..<dot]
        guard !num.isEmpty, num.allSatisfy(\.isNumber) else { return false }
        let after = s.index(after: dot)
        return after < s.endIndex && s[after] == " "
    }

    private static func stripOrdered(_ s: String) -> String {
        guard let dot = s.firstIndex(of: ".") else { return s }
        return String(s[s.index(after: dot)...]).trimmingCharacters(in: .whitespaces)
    }
}
