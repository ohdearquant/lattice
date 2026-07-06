// Pack-list guard: asserts the main package tarball (`npm pack --dry-run
// --json`, piped in on stdin) contains the JS-facing surface but none of
// the Rust build inputs or the platform-specific `.node` binary -- those
// ship only in the per-platform `@khive-ai/lattice-embed-<target>` packages
// (npm/<target>/package.json), never in this package. Run via `npm run
// packlist` (see package.json).
import assert from 'node:assert/strict'

let input = ''
for await (const chunk of process.stdin) {
  input += chunk
}

if (!input.trim()) {
  throw new Error('expected npm pack --dry-run --json input on stdin')
}

const parsed = JSON.parse(input)
const pack = Array.isArray(parsed) ? parsed[0] : parsed
const files = (pack.files || []).map(file => file.path)

const required = [
  'package.json',
  'README.md',
  'binding.js',
  'index.js',
  'index.d.ts'
]

for (const path of required) {
  assert.ok(files.includes(path), `main package is missing ${path}`)
}

const forbiddenPatterns = [
  /^target\//,
  /^src\//,
  /\.node$/,
  /^npm\//,
  /^\.github\//,
  /^__test__\//,
  /^bench\//,
  /^examples\//,
  /^Cargo\.toml$/,
  /^Cargo\.lock$/,
  /^build\.rs$/,
  /^scripts\//
]

for (const path of files) {
  for (const pattern of forbiddenPatterns) {
    assert.ok(!pattern.test(path), `main package must not include ${path}`)
  }
}

console.log(JSON.stringify({
  ok: true,
  package: pack.name,
  version: pack.version,
  fileCount: files.length
}, null, 2))
