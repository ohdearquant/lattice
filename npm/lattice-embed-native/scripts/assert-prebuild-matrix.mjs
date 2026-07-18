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

const workflowTargets = Array.from(
  workflow.matchAll(/^\s+-?\s*target:\s+([a-z0-9_-]+)\s*$/gm),
  match => match[1]
)

assert.deepEqual(
  new Set(workflowTargets),
  new Set(targetPackages.keys()),
  'the prebuild workflow must build every supported napi target'
)
assert.equal(
  workflowTargets.length,
  targetPackages.size,
  'the prebuild workflow must contain exactly one build row per supported napi target'
)

for (const command of [
  'if-no-files-found: error',
  'npm run create-npm-dirs',
  'npm run artifacts',
  'make publish-npm-dry',
  'make publish-npm'
]) {
  assert.ok(workflow.includes(command), `the prebuild workflow is missing: ${command}`)
}

console.log(JSON.stringify({ ok: true, targets: workflowTargets }, null, 2))
