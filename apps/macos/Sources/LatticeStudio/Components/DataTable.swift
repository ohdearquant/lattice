import SwiftUI

// MARK: - Primitive 5: DataTable
//
// Generic, column-descriptor-driven table.
//
// Design spec:
//   - Rows at Theme.Space.rowHeight (28px) or rowHeightComfortable (32px)
//   - Tabular-mono right-aligned numerals
//   - 1px hairline row rules (no zebra)
//   - Sortable 11pt all-caps headers (tap header to sort; second tap reverses)
//   - Selected row: 2px teal left-border accent (NO fill flood)
//   - No external borders — table is "full-bleed" within its panel

/// Alignment role for a column's cell content.
enum ColumnAlignment {
    case leading    // text: labels, names
    case trailing   // numerals: sizes, counts, metrics (tabular-mono)
    case center
}

/// Descriptor for a single table column.
struct ColumnDef<Row> {
    let id: String
    let header: String
    let alignment: ColumnAlignment
    let minWidth: CGFloat
    let value: (Row) -> String
    let isNumeric: Bool

    init(
        id: String,
        header: String,
        alignment: ColumnAlignment = .trailing,
        minWidth: CGFloat = 80,
        isNumeric: Bool = true,
        value: @escaping (Row) -> String
    ) {
        self.id = id
        self.header = header
        self.alignment = alignment
        self.minWidth = minWidth
        self.isNumeric = isNumeric
        self.value = value
    }
}

/// Generic instrument data table.
///
/// ```swift
/// DataTable(
///     rows: store.models,
///     columns: [
///         ColumnDef(id:"name", header:"MODEL", alignment:.leading, isNumeric:false) { $0.name },
///         ColumnDef(id:"size", header:"SIZE GB") { String(format:"%.2f", Double($0.sizeBytes)/1e9) },
///     ],
///     selectedID: $selectedID,
///     comfortable: store.rowComfortable
/// )
/// ```
struct DataTable<Row: Identifiable>: View where Row.ID: Hashable {
    let rows: [Row]
    let columns: [ColumnDef<Row>]
    @Binding var selectedID: Row.ID?
    var comfortable: Bool = false

    // Sort state
    @State private var sortColumnID: String? = nil
    @State private var sortAscending: Bool = true

    private var rowHeight: CGFloat {
        comfortable ? Theme.Space.rowHeightComfortable : Theme.Space.rowHeight
    }

    private var sortedRows: [Row] {
        guard let colID = sortColumnID,
              let col = columns.first(where: { $0.id == colID })
        else { return rows }

        return rows.sorted { a, b in
            let va = col.value(a)
            let vb = col.value(b)
            // Numeric sort when column is numeric
            if col.isNumeric, let da = Double(va), let db = Double(vb) {
                return sortAscending ? da < db : da > db
            }
            return sortAscending ? va < vb : va > vb
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header row
            headerRow

            // Data rows
            ForEach(Array(sortedRows.enumerated()), id: \.element.id) { _, row in
                dataRow(row: row)
            }
        }
    }

    private var headerRow: some View {
        HStack(spacing: 0) {
            ForEach(columns, id: \.id) { col in
                Button {
                    if sortColumnID == col.id {
                        sortAscending.toggle()
                    } else {
                        sortColumnID = col.id
                        sortAscending = true
                    }
                } label: {
                    HStack(spacing: 2) {
                        Text(col.header)
                            .instrumentLabel()
                            .frame(maxWidth: .infinity, alignment: frameAlignment(col.alignment))

                        // Sort indicator
                        if sortColumnID == col.id {
                            Text(sortAscending ? "↑" : "↓")
                                .font(Theme.Fonts.cell)
                                .foregroundStyle(Theme.Palette.signal)
                        }
                    }
                    .padding(.horizontal, Theme.Space.sm)
                    .frame(minWidth: col.minWidth, maxWidth: .infinity, alignment: frameAlignment(col.alignment))
                }
                .buttonStyle(.plain)
                .frame(minWidth: col.minWidth, maxWidth: .infinity)
            }
        }
        .frame(height: rowHeight)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }

    private func dataRow(row: Row) -> some View {
        let isSelected = selectedID == row.id

        return HStack(spacing: 0) {
            // 2px teal left-border accent on selected row (no fill flood)
            Rectangle()
                .fill(isSelected ? Theme.Palette.signal : .clear)
                .frame(width: 2)

            ForEach(columns, id: \.id) { col in
                let cellValue = col.value(row)
                Text(cellValue)
                    .font(col.isNumeric ? Theme.Fonts.cell : Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.ink)
                    .monospacedDigit()
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .padding(.horizontal, Theme.Space.sm)
                    .frame(minWidth: col.minWidth, maxWidth: .infinity, alignment: frameAlignment(col.alignment))
            }
        }
        .frame(height: rowHeight)
        .contentShape(Rectangle())
        .onTapGesture {
            selectedID = isSelected ? nil : row.id
        }
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }

    private func frameAlignment(_ col: ColumnAlignment) -> Alignment {
        switch col {
        case .leading: .leading
        case .trailing: .trailing
        case .center: .center
        }
    }
}

// MARK: - Previews

#Preview("DataTable") {
    struct SampleRow: Identifiable {
        let id: String
        let name: String
        let params: String
        let format: String
        let sizeGB: String
        let files: String
    }

    let rows: [SampleRow] = [
        SampleRow(id: "1", name: "qwen3.5-0.8b", params: "0.8B", format: "BF16", sizeGB: "1.61", files: "4"),
        SampleRow(id: "2", name: "qwen3.5-0.8b-q4", params: "0.8B", format: "Q4", sizeGB: "0.41", files: "3"),
        SampleRow(id: "3", name: "qwen3.5-0.8b-quarot", params: "0.8B", format: "QuaRot Q4", sizeGB: "0.40", files: "5"),
        SampleRow(id: "4", name: "qwen3.5-2b", params: "2B", format: "BF16", sizeGB: "4.07", files: "4"),
    ]

    @State var selected: String? = "1"

    return DataTable(
        rows: rows,
        columns: [
            ColumnDef(id: "name", header: "MODEL", alignment: .leading, minWidth: 180, isNumeric: false) { $0.name },
            ColumnDef(id: "params", header: "PARAMS", minWidth: 60) { $0.params },
            ColumnDef(id: "format", header: "FORMAT", alignment: .leading, minWidth: 80, isNumeric: false) { $0.format },
            ColumnDef(id: "size", header: "SIZE GB", minWidth: 70) { $0.sizeGB },
            ColumnDef(id: "files", header: "FILES", minWidth: 50) { $0.files },
        ],
        selectedID: Binding(get: { selected }, set: { selected = $0 }),
        comfortable: false
    )
    .instrumentPanel()
    .background(Theme.Palette.canvas)
    .frame(width: 560, height: 200)
}
