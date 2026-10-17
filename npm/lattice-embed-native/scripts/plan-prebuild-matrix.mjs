// Plans the native prebuild matrix for one workflow event and prints it as
// GITHUB_OUTPUT lines. A pull request builds only the rows marked
// `pull_request`; push and workflow_dispatch build every row, so a release is
// always assembled from the full matrix. The package job checks exactly the
// suffixes planned here.
import { readFile } from 'node:fs/promises'

const rows = JSON.parse(await readFile(new URL('./prebuild-matrix.json', import.meta.url), 'utf8'))
const event = process.env.EVENT
if (!event) {
  throw new Error('EVENT must name the workflow event')
}

const planned = event === 'pull_request' ? rows.filter(row => row.pull_request) : rows
if (planned.length === 0) {
  throw new Error(`no prebuild targets planned for event ${event}`)
}

const include = planned.map(({ runner, target, args }) => ({ runner, target, args }))
console.log(`include=${JSON.stringify(include)}`)
console.log(`suffixes=${planned.map(row => row.suffix).join(' ')}`)
// `napi artifacts` refuses when any configured target lacks a binary, so the
// package job hands it a config naming only the planned targets.
console.log(`napi_config=${JSON.stringify({ targets: planned.map(row => row.target) })}`)
console.error(
  `planned ${planned.length} of ${rows.length} prebuild targets for ${event}: ` +
    planned.map(row => row.target).join(', ')
)
