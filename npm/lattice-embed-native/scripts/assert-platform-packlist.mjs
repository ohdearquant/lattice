// Pack-list guard for a per-platform binary subpackage (e.g.
// `npm/darwin-arm64/`): asserts the tarball (`npm pack --dry-run --json`,
// piped in on stdin) contains `package.json` plus EXACTLY ONE `.node` file,
// and nothing else. Run from the main package root via `npm run
// packlist:darwin-arm64` (see package.json), which `cd`s into the
// subpackage directory first. The `napi artifacts` step (`npm run
// artifacts`) must have already copied the built `.node` binary into the
// subpackage directory before this check is meaningful -- an empty
// subpackage tarball (just `package.json`) is exactly the defect this
// script exists to catch.
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

assert.ok(files.includes('package.json'), `platform package is missing package.json (got: ${files.join(', ')})`)

const nodeFiles = files.filter(path => path.endsWith('.node'))
assert.equal(
  nodeFiles.length,
  1,
  `platform package must ship exactly one .node file, found ${nodeFiles.length} (${nodeFiles.join(', ') || 'none'}); ` +
    'run `npm run artifacts` to copy the built binary into this subpackage before packing'
)

assert.equal(
  files.length,
  2,
  `platform package must contain only package.json and one .node file, found: ${files.join(', ')}`
)

console.log(JSON.stringify({
  ok: true,
  package: pack.name,
  version: pack.version,
  fileCount: files.length,
  nodeFile: nodeFiles[0]
}, null, 2))
