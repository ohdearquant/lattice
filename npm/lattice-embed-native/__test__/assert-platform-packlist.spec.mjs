// Node test-runner suite for scripts/assert-platform-packlist.mjs. Builds
// synthetic platform-package fixtures under a scratch tmpdir, packs each
// with the real `npm pack --dry-run --json` (no network, no publish), and
// pipes the output through the guard script as a child process -- exactly
// how the prebuild workflow invokes it via `packlist:<suffix>`.
import assert from 'node:assert/strict'
import test, { after } from 'node:test'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const SCRIPT = join(dirname(fileURLToPath(import.meta.url)), '..', 'scripts', 'assert-platform-packlist.mjs')

// A per-run cache this suite owns outright, so the real `npm pack` calls
// below never depend on who owns (or whether anyone can write to) the
// ambient ~/.npm cache -- e.g. a sandboxed process without permission to
// touch ~/.npm/_cacache.
const NPM_CACHE_DIR = mkdtempSync(join(tmpdir(), 'packlist-npm-cache-'))
after(() => {
  rmSync(NPM_CACHE_DIR, { recursive: true, force: true })
})

function makeFixture(files) {
  const dir = mkdtempSync(join(tmpdir(), 'packlist-fixture-'))
  for (const [path, content] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, path)), { recursive: true })
    writeFileSync(join(dir, path), content)
  }
  return dir
}

function packAndAssert(dir) {
  const packOutput = execFileSync('npm', ['pack', '--dry-run', '--json'], {
    cwd: dir,
    encoding: 'utf8',
    env: { ...process.env, NPM_CONFIG_CACHE: NPM_CACHE_DIR },
  })
  execFileSync('node', [SCRIPT], { input: packOutput, encoding: 'utf8' })
}

const basePackageJson = (files) => JSON.stringify({ name: 'test-platform-pkg', version: '1.0.0', files }, null, 2)

test('accepts a single custom-extension root README', () => {
  const dir = makeFixture({
    'package.json': basePackageJson(['test.node']),
    'test.node': '',
    'README.custom': 'custom readme\n',
  })
  try {
    assert.doesNotThrow(() => packAndAssert(dir))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('rejects two root README-like entries', () => {
  const dir = makeFixture({
    'package.json': basePackageJson(['test.node']),
    'test.node': '',
    'README.md': 'readme md\n',
    'README.txt': 'readme txt\n',
  })
  try {
    assert.throws(() => packAndAssert(dir))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('rejects an allowlisted readme-prefixed non-README file', () => {
  const dir = makeFixture({
    'package.json': basePackageJson(['test.node', 'readme-not-a-readme.js']),
    'test.node': '',
    'readme-not-a-readme.js': 'not a readme\n',
  })
  try {
    assert.throws(() => packAndAssert(dir))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('accepts a multi-dot root README (npm auto-includes README.md.bak)', () => {
  const dir = makeFixture({
    'package.json': basePackageJson(['test.node']),
    'test.node': '',
    'README.md.bak': 'multi-dot readme\n',
  })
  try {
    assert.doesNotThrow(() => packAndAssert(dir))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('a root README ending in ~ is excluded by npm itself, not by this guard', () => {
  // Empirically verified via `npm pack --dry-run --json`: npm-packlist's own grammar
  // (`!/readme{,.*[^~$]}`) excludes trailing-~/$ variants from the tarball entirely, so
  // README.md~ never reaches the guard's file list -- the pack succeeds as if it were absent.
  const dir = makeFixture({
    'package.json': basePackageJson(['test.node']),
    'test.node': '',
    'README.md~': 'trailing tilde readme\n',
  })
  try {
    assert.doesNotThrow(() => packAndAssert(dir))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('rejects an ordinary unexpected extra file', () => {
  const dir = makeFixture({
    'package.json': basePackageJson(['test.node', 'extra.txt']),
    'test.node': '',
    'extra.txt': 'extra\n',
  })
  try {
    assert.throws(() => packAndAssert(dir))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('rejects empty stdin', () => {
  assert.throws(() => execFileSync('node', [SCRIPT], { input: '', encoding: 'utf8' }))
})

test('rejects a payload file under a README-named directory', () => {
  // README.docs/ is a directory, not the root README file npm auto-includes. A
  // dot-star matcher would treat 'README.docs/payload.txt' as an exempt README
  // (crossing the '/'); a segment matcher must not.
  const dir = makeFixture({
    'package.json': basePackageJson(['test.node', 'README.docs']),
    'test.node': '',
    'README.docs/payload.txt': 'arbitrary payload smuggled via a README-named dir\n',
  })
  try {
    assert.throws(() => packAndAssert(dir))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('accepts a root README whose suffix contains a line terminator', () => {
  // JavaScript `.` does not match line terminators, but minimatch `*` does not
  // exclude them either -- npm itself includes this file, so the guard must too.
  const dir = makeFixture({
    'package.json': basePackageJson(['test.node']),
    'test.node': '',
    'README.\nnotes': 'line-terminator suffix readme\n',
  })
  try {
    assert.doesNotThrow(() => packAndAssert(dir))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('accepts a bare README with no extension', () => {
  const dir = makeFixture({
    'package.json': basePackageJson(['test.node']),
    'test.node': '',
    README: 'bare readme, no extension\n',
  })
  try {
    assert.doesNotThrow(() => packAndAssert(dir))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
