// swift-tools-version: 6.0
import PackageDescription

// Lattice Studio — the instrument panel for the pure-Rust lattice engine.
// Driven entirely via lattice CLI subprocesses (line-delimited `@@lattice {json}`
// event protocol). No in-process ML, no Python, no ONNX — same posture as the engine.
let package = Package(
    name: "LatticeStudio",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "LatticeStudio", targets: ["LatticeStudio"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "LatticeStudio",
            path: "Sources/LatticeStudio",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        ),
        .testTarget(
            name: "LatticeStudioTests",
            dependencies: ["LatticeStudio"],
            path: "Tests/LatticeStudioTests",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        )
    ]
)
