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
import { fileURLToPath } from 'node:url'

const nativeDir = dirname(dirname(fileURLToPath(import.meta.url)))
const pkg = JSON.parse(readFileSync(join(nativeDir, 'package.json'), 'utf8'))
const platforms = Object.keys(pkg.optionalDependencies || {}).map(name =>
  name.replace('@khive-ai/lattice-embed-', '')
)

function platformBinariesPresent() {
  if (platforms.length === 0) return false
  for (const platform of platforms) {
    const platformPkgPath = join(nativeDir, 'npm', platform, 'package.json')
    if (!existsSync(platformPkgPath)) return false
    const platformPkg = JSON.parse(readFileSync(platformPkgPath, 'utf8'))
    const main = platformPkg.main || ''
    if (!main || !existsSync(join(nativeDir, 'npm', platform, main))) return false
  }
  return true
}

if (platformBinariesPresent()) {
  console.log(
    'Platform prebuilds already present for every configured target; skipping `napi artifacts`.'
  )
} else {
  execFileSync('npm', ['run', 'artifacts'], { cwd: nativeDir, stdio: 'inherit' })
}
