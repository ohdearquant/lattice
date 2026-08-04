// Pack-list guard for a per-platform binary subpackage (e.g.
// `npm/darwin-arm64/`): asserts the tarball (`npm pack --dry-run --json`,
// piped in on stdin) contains `package.json`, EXACTLY ONE `.node` file, and
// nothing else besides an npm-auto-included README (npm always bundles the
// root README file itself -- case- and extension-flexible, e.g. `README`,
// `README.md`, `readme.custom` -- regardless of the package.json "files"
// allowlist, so a two-file expectation is wrong whenever one is present).
// This does NOT extend to arbitrary README-prefixed files: an allowlisted
// `readme-not-a-readme.js` is not npm's special case and must still be
// rejected.
// Run from the main package root via `npm run packlist:darwin-arm64` (see
// package.json), which `cd`s into the subpackage directory first, or from
// the prebuild workflow's per-suffix loop. The `napi artifacts` step (`npm
// run artifacts`) must have already copied the built `.node` binary into
// the subpackage directory before this check is meaningful -- an empty
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

const allowed = new Set(['package.json', nodeFiles[0]])
const readmePattern = /^README(\.[A-Za-z0-9]+)?$/i
const readmeFiles = files.filter(path => !allowed.has(path) && readmePattern.test(path))
assert.ok(
  readmeFiles.length <= 1,
  `platform package must contain at most one root README file, found ${readmeFiles.length} (${readmeFiles.join(', ')})`
)

const unexpected = files.filter(path => !allowed.has(path) && !readmePattern.test(path))
assert.equal(
  unexpected.length,
  0,
  `platform package must contain only package.json, one .node file, and an optional README; ` +
    `found unexpected entries: ${unexpected.join(', ')} (full list: ${files.join(', ')})`
)

console.log(JSON.stringify({
  ok: true,
  package: pack.name,
  version: pack.version,
  fileCount: files.length,
  nodeFile: nodeFiles[0]
}, null, 2))
