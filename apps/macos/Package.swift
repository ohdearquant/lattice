// swift-tools-version: 6.0
import PackageDescription

// Lattice — the instrument panel for the pure-Rust lattice engine.
// Driven entirely via lattice CLI subprocesses (line-delimited `@@lattice {json}`
// event protocol). No in-process ML, no Python, no ONNX — same posture as the engine.
//
// The SwiftPM product is "Lattice" (the user-facing app name); the source directory keeps its
// historical name `Sources/LatticeStudio` via the explicit `path:` so the rename touches no files.
let package = Package(
    name: "Lattice",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "Lattice", targets: ["Lattice"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "Lattice",
            path: "Sources/LatticeStudio",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        ),
        .testTarget(
            name: "LatticeStudioTests",
            dependencies: ["Lattice"],
            path: "Tests/LatticeStudioTests",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        )
    ]
)
