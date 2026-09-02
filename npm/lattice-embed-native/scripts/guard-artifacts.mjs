// `napi artifacts --output-dir . --npm-dir npm` (the package.json "artifacts"
// script) reconciles --npm-dir against --output-dir: for every configured
// target it looks for a freshly built binary under --output-dir, and when
// one platform's --output-dir binary is absent it deletes that platform's
// existing --npm-dir binary rather than leaving it alone, then exits nonzero
// once nothing is left to place. That is correct when npm/<platform>/ is
// still empty and this step's job is to collect a fresh local build into it,
// and destructive when every platform binary has already been placed there
// by another means (e.g. downloaded prebuilt binaries) with no matching
// fresh build sitting in --output-dir to reconcile against. This guard runs
// before prepublishOnly's `napi artifacts` invocation and skips it entirely
// when every platform in optionalDependencies already resolves its exact
// `main`-named .node file, so a release whose platform packages are already
// fully populated is not silently emptied out during `npm publish`.
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const nativeDir = dirname(dirname(fileURLToPath(import.meta.url)))

// A platform package's "main" is trusted only when it resolves to exactly
// the .node file the platform directory ships -- not any existing path the
// field happens to name. Rejects an absent/empty value, a path separator
// (forward or back), a ".." segment, and any extension other than ".node".
export function isValidPlatformMain(main) {
  return Boolean(main) && !main.includes('/') && !main.includes('\\') && !main.includes('..') && main.endsWith('.node')
}

// npm's own executable is `npm.cmd` on Windows; spawning the bare `npm`
// name via execFileSync (no shell) throws ENOENT there instead of running
// the artifacts step.
export function npmCommandFor(platform) {
  return platform === 'win32' ? 'npm.cmd' : 'npm'
}

export function platformBinariesPresent(dir) {
  const pkg = JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8'))
  const platforms = Object.keys(pkg.optionalDependencies || {}).map(name =>
    name.replace('@khive-ai/lattice-embed-', '')
  )
  if (platforms.length === 0) return false
  for (const platform of platforms) {
    const platformPkgPath = join(dir, 'npm', platform, 'package.json')
    if (!existsSync(platformPkgPath)) return false
    const platformPkg = JSON.parse(readFileSync(platformPkgPath, 'utf8'))
    const main = platformPkg.main || ''
    if (!isValidPlatformMain(main) || !existsSync(join(dir, 'npm', platform, main))) return false
  }
  return true
}

const isMainModule = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href
if (isMainModule) {
  if (platformBinariesPresent(nativeDir)) {
    console.log(
      'Platform prebuilds already present for every configured target; skipping `napi artifacts`.'
    )
  } else {
    execFileSync(npmCommandFor(process.platform), ['run', 'artifacts'], { cwd: nativeDir, stdio: 'inherit' })
  }
}
