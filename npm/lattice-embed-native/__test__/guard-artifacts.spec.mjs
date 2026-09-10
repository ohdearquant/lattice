// Node test-runner suite for scripts/guard-artifacts.mjs's two exported
// helpers: platformBinariesPresent (does every platform in
// optionalDependencies already carry a trustworthy, present .node binary)
// and npmCommandFor (which npm executable name to spawn on the current
// platform). Fixtures build a scratch native-package tree under a temp
// dir; nothing here touches the real npm/ layout or spawns a real build.
import assert from 'node:assert/strict'
import test from 'node:test'
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { platformBinariesPresent, npmCommandFor } from '../scripts/guard-artifacts.mjs'

const NAME_PREFIX = '@khive-ai/lattice-embed-'

function makeFixture(files) {
  const dir = mkdtempSync(join(tmpdir(), 'guard-artifacts-fixture-'))
  for (const [path, content] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, path)), { recursive: true })
    writeFileSync(join(dir, path), content)
  }
  return dir
}

function nativePackageJson(platforms) {
  const optionalDependencies = {}
  for (const platform of platforms) optionalDependencies[`${NAME_PREFIX}${platform}`] = '1.0.0'
  return JSON.stringify({ name: 'test-native-pkg', version: '1.0.0', optionalDependencies }, null, 2)
}

function platformPackageJson(platform, main) {
  return JSON.stringify({ name: `${NAME_PREFIX}${platform}`, version: '1.0.0', main }, null, 2)
}

test('all platforms populated with a valid .node main returns true', () => {
  const dir = makeFixture({
    'package.json': nativePackageJson(['darwin-arm64', 'linux-x64-gnu']),
    'npm/darwin-arm64/package.json': platformPackageJson('darwin-arm64', 'lattice-embed-native.darwin-arm64.node'),
    'npm/darwin-arm64/lattice-embed-native.darwin-arm64.node': 'fake-binary',
    'npm/linux-x64-gnu/package.json': platformPackageJson('linux-x64-gnu', 'lattice-embed-native.linux-x64-gnu.node'),
    'npm/linux-x64-gnu/lattice-embed-native.linux-x64-gnu.node': 'fake-binary',
  })
  try {
    assert.equal(platformBinariesPresent(dir), true)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('a platform whose main is missing returns false', () => {
  const dir = makeFixture({
    'package.json': nativePackageJson(['darwin-arm64']),
    // "main" omitted entirely.
    'npm/darwin-arm64/package.json': JSON.stringify({ name: `${NAME_PREFIX}darwin-arm64`, version: '1.0.0' }),
  })
  try {
    assert.equal(platformBinariesPresent(dir), false)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('a platform whose main is empty returns false', () => {
  const dir = makeFixture({
    'package.json': nativePackageJson(['darwin-arm64']),
    'npm/darwin-arm64/package.json': platformPackageJson('darwin-arm64', ''),
  })
  try {
    assert.equal(platformBinariesPresent(dir), false)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('a platform whose main is ../evil.js returns false', () => {
  // The escaped file is created and DOES exist on disk, so a false "true"
  // here would prove the rejection is a missing-file coincidence rather
  // than the main-format validation this test targets.
  const dir = makeFixture({
    'package.json': nativePackageJson(['darwin-arm64']),
    'npm/darwin-arm64/package.json': platformPackageJson('darwin-arm64', '../evil.js'),
    'npm/evil.js': 'arbitrary file escaping the platform directory\n',
  })
  try {
    assert.equal(platformBinariesPresent(dir), false)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('a platform whose main is README.md returns false', () => {
  const dir = makeFixture({
    'package.json': nativePackageJson(['darwin-arm64']),
    'npm/darwin-arm64/package.json': platformPackageJson('darwin-arm64', 'README.md'),
    'npm/darwin-arm64/README.md': '# not a binary\n',
  })
  try {
    assert.equal(platformBinariesPresent(dir), false)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('npm command resolves to npm.cmd on win32 and npm elsewhere', () => {
  assert.equal(npmCommandFor('win32'), 'npm.cmd')
  assert.equal(npmCommandFor('darwin'), 'npm')
  assert.equal(npmCommandFor('linux'), 'npm')
})
