import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'))
const workflow = await readFile(
  new URL('../../../.github/workflows/npm-prebuild.yml', import.meta.url),
  'utf8'
)

const targetPackages = new Map([
  ['aarch64-apple-darwin', '@khive-ai/lattice-embed-darwin-arm64'],
  ['x86_64-apple-darwin', '@khive-ai/lattice-embed-darwin-x64'],
  ['x86_64-unknown-linux-gnu', '@khive-ai/lattice-embed-linux-x64-gnu'],
  ['x86_64-unknown-linux-musl', '@khive-ai/lattice-embed-linux-x64-musl'],
  ['aarch64-unknown-linux-gnu', '@khive-ai/lattice-embed-linux-arm64-gnu'],
  ['aarch64-unknown-linux-musl', '@khive-ai/lattice-embed-linux-arm64-musl'],
  ['x86_64-pc-windows-msvc', '@khive-ai/lattice-embed-win32-x64-msvc']
])

assert.deepEqual(
  new Set(packageJson.napi.targets),
  new Set(targetPackages.keys()),
  'napi.targets must match the supported native prebuild matrix'
)

assert.deepEqual(
  new Set(Object.keys(packageJson.optionalDependencies)),
  new Set(targetPackages.values()),
  'optionalDependencies must match the supported native prebuild packages'
)

const rows = JSON.parse(await readFile(new URL('./prebuild-matrix.json', import.meta.url), 'utf8'))
const matrixTargets = rows.map(row => row.target)

assert.deepEqual(
  new Set(matrixTargets),
  new Set(targetPackages.keys()),
  'prebuild-matrix.json must build every supported napi target'
)
assert.equal(
  matrixTargets.length,
  targetPackages.size,
  'prebuild-matrix.json must contain exactly one row per supported napi target'
)

for (const row of rows) {
  assert.equal(
    row.suffix,
    targetPackages.get(row.target).replace('@khive-ai/lattice-embed-', ''),
    `prebuild-matrix.json row ${row.target} must name its platform package suffix`
  )
  assert.ok(typeof row.runner === 'string' && row.runner.length > 0, `row ${row.target} needs a runner`)
  assert.equal(typeof row.args, 'string', `row ${row.target} needs an args string`)
  assert.equal(typeof row.pull_request, 'boolean', `row ${row.target} needs a pull_request boolean`)
}
assert.ok(
  rows.some(row => row.pull_request),
  'at least one prebuild target must build on pull requests'
)

// The build matrix and the package-assembly loop must both come from the plan:
// a static `target:` row or a literal suffix loop would bypass it.
assert.equal(
  Array.from(workflow.matchAll(/^\s+-?\s*target:\s+([a-z0-9_-]+)\s*$/gm)).length,
  0,
  'the prebuild workflow must take its build rows from the plan, not static target rows'
)
for (const wiring of [
  'scripts/plan-prebuild-matrix.mjs',
  'include: ${{ fromJSON(needs.plan.outputs.include) }}',
  'PLANNED_SUFFIXES: ${{ needs.plan.outputs.suffixes }}',
  'for suffix in $PLANNED_SUFFIXES; do',
  'PLANNED_NAPI_CONFIG: ${{ needs.plan.outputs.napi_config }}',
  'npm run artifacts -- --config-path "$RUNNER_TEMP/napi-planned.json"'
]) {
  assert.ok(workflow.includes(wiring), `the prebuild workflow is missing: ${wiring}`)
}

for (const command of [
  'if-no-files-found: error',
  'npm run create-npm-dirs',
  'npm run artifacts',
  'TARGET_CC: clang',
  'TARGET_CXX: clang++',
  'make publish-npm-dry',
  'make publish-npm'
]) {
  assert.ok(workflow.includes(command), `the prebuild workflow is missing: ${command}`)
}

console.log(
  JSON.stringify(
    {
      ok: true,
      targets: matrixTargets,
      pull_request: rows.filter(row => row.pull_request).map(row => row.target)
    },
    null,
    2
  )
)
