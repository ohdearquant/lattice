'use strict'

const { existsSync, readFileSync } = require('node:fs')
const { join } = require('node:path')
const { execFileSync } = require('node:child_process')

const BINARY_NAME = 'lattice-embed-native'
const PACKAGE_PREFIX = '@lattice-embed/native'
const loadErrors = []

function requireCandidate(candidate) {
  try {
    return require(candidate)
  } catch (error) {
    loadErrors.push({ candidate, error })
    return null
  }
}

function requireLocal(suffix) {
  const candidates = [
    join(__dirname, `${BINARY_NAME}.${suffix}.node`),
    join(__dirname, `${BINARY_NAME}.node`)
  ]

  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      const binding = requireCandidate(candidate)
      if (binding) return binding
    }
  }

  return null
}

function requirePackage(suffix) {
  return requireCandidate(`${PACKAGE_PREFIX}-${suffix}`)
}

function isMusl() {
  if (process.platform !== 'linux') return false

  const report = process.report && typeof process.report.getReport === 'function'
    ? process.report.getReport()
    : null

  if (report && report.header && report.header.glibcVersionRuntime) {
    return false
  }

  try {
    const output = execFileSync('ldd', ['--version'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe']
    })
    return output.toLowerCase().includes('musl')
  } catch (_) {
    // Some musl systems print version information to stderr or do not support --version.
  }

  try {
    const ldd = readFileSync('/usr/bin/ldd', 'utf8')
    return ldd.toLowerCase().includes('musl')
  } catch (_) {
    return false
  }
}

function platformSuffix() {
  const platform = process.platform
  const arch = process.arch

  if (platform === 'darwin') {
    if (arch === 'arm64') return 'darwin-arm64'
    if (arch === 'x64') return 'darwin-x64'
  }

  if (platform === 'win32') {
    if (arch === 'x64') return 'win32-x64-msvc'
  }

  if (platform === 'linux') {
    const libc = isMusl() ? 'musl' : 'gnu'
    if (arch === 'x64') return `linux-x64-${libc}`
    if (arch === 'arm64') return `linux-arm64-${libc}`
  }

  return null
}

function loadNative() {
  const suffix = platformSuffix()

  if (!suffix) {
    const error = new Error(
      `FL_EMBED_UNSUPPORTED_PLATFORM: unsupported platform for ${PACKAGE_PREFIX}: ` +
      `platform=${process.platform}, arch=${process.arch}, libc=${process.platform === 'linux' ? (isMusl() ? 'musl' : 'gnu') : 'n/a'}`
    )
    error.code = 'FL_EMBED_UNSUPPORTED_PLATFORM'
    throw error
  }

  const local = requireLocal(suffix)
  if (local) return local

  const packaged = requirePackage(suffix)
  if (packaged) return packaged

  const details = loadErrors
    .map(({ candidate, error }) => `- ${candidate}: ${error && error.message ? error.message : String(error)}`)
    .join('\n')

  const error = new Error(
    `FL_EMBED_NATIVE_LOAD_FAILED: unable to load native binding for ${PACKAGE_PREFIX}\n` +
    `platform=${process.platform}, arch=${process.arch}, suffix=${suffix}\n` +
    `Reinstall with optional dependencies enabled. This package intentionally has no default Rust source-build fallback.\n` +
    details
  )
  error.code = 'FL_EMBED_NATIVE_LOAD_FAILED'
  throw error
}

module.exports = loadNative()
