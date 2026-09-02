// Builds viewer.html: one self-contained page showing every workflow in this
// directory from above -- phases, agents, fan-out, prompts, outcome.
//
//   node .claude/workflows/viz.mjs
//
// It does no parsing of its own. dryrun.mjs already executes each workflow with
// stubbed runtime globals, so every prompt comes out fully interpolated and the
// call trace is real; this reads what dryrun writes (calls.json, the NN-*.txt
// prompts, result.json) and inlines it into HTML. Re-run after editing any
// workflow -- nothing here notices staleness on its own.
import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'))

// Three of the four refuse to run without args and would render as an empty
// column. These are the smallest arg sets that reach every stage.
const WORKFLOWS = [
  { name: 'process-issues', args: null },
  { name: 'prepare-tickets', args: { request: 300 } },
  { name: 'express-lane', args: { issues: [201, 204], merge: true } },
  { name: 'run-merge-train', args: { spec: 201, merge: true } },
]

// meta is guaranteed a pure literal -- dryrun.mjs fails the run otherwise -- so
// slicing and evaluating the block is safe, and beats a second parser.
const readMeta = (src) => {
  const start = src.indexOf('export const meta')
  const end = src.slice(start).search(/\r?\n\}\r?\n/)
  const block = src.slice(start, start + end + 2).replace('export const meta', 'meta')
  return new Function(`let meta; ${block}; return meta`)()
}

const collect = ({ name, args }) => {
  const outDir = path.join(here, `.dryrun-${name}`)
  const run = spawnSync(process.execPath, [path.join(here, 'dryrun.mjs'), name], {
    env: { ...process.env, DRYRUN_ARGS: args ? JSON.stringify(args) : '' },
    encoding: 'utf8',
  })
  if (run.status !== 0) throw new Error(`dryrun ${name} failed:\n${run.stdout}${run.stderr}`)

  const { calls, workflowCalls, phases } = JSON.parse(fs.readFileSync(path.join(outDir, 'calls.json'), 'utf8'))
  const { result, logs } = JSON.parse(fs.readFileSync(path.join(outDir, 'result.json'), 'utf8'))

  // Same filename dryrun.mjs:412 builds, rather than trusting a directory sort.
  const withPrompts = calls.map((c) => ({
    ...c,
    prompt: fs.readFileSync(
      path.join(outDir, `${String(c.n).padStart(2, '0')}-${c.label.replace(/[^\w.#-]/g, '_')}.txt`),
      'utf8',
    ),
  }))

  const srcPath = path.join(here, `${name}.js`)
  const meta = readMeta(fs.readFileSync(srcPath, 'utf8'))
  // A file:// page cannot stat the filesystem, so it cannot detect staleness itself.
  // Stamping both times is the most it can honestly say: compare them by eye.
  const edited = fs.statSync(srcPath).mtime.toISOString().slice(0, 16).replace('T', ' ')
  return { ...meta, args, edited, calls: withPrompts, workflowCalls, topPhases: phases, result, logs }
}

const data = WORKFLOWS.map(collect)

// -- self-check -------------------------------------------------------------
// A workflow that starts refusing its sample args must break the generator, not
// quietly render an empty column.
const problems = []
for (const w of data) {
  if (!w.calls.length) problems.push(`${w.name}: dryrun produced no agent calls -- its sample args are stale`)
  if (w.result && typeof w.result === 'object' && w.result.error) problems.push(`${w.name}: ${w.result.error}`)
  for (const c of w.calls) {
    if (!w.phases.some((p) => p.title === c.phase)) problems.push(`${w.name}: ${c.label} is in phase "${c.phase}", absent from meta.phases`)
  }
}
if (problems.length) {
  console.error(`viz: ${problems.length} problem(s)`)
  for (const p of problems) console.error(`  - ${p}`)
  process.exit(1)
}

const out = path.join(here, 'viewer.html')
fs.writeFileSync(out, render(data, new Date().toISOString().slice(0, 16).replace('T', ' ')))

for (const w of data) {
  console.log(`  ${String(w.calls.length).padStart(3)} agents  ${String(w.phases.length).padStart(2)} phases  ${w.name}`)
}
console.log(`\nwrote ${out} (${(fs.statSync(out).size / 1024).toFixed(0)} KB) -- open it in a browser`)

// -- the page ---------------------------------------------------------------
// app() is stringified into the page rather than written as a template string:
// its own template literals would otherwise be interpolated here, by node.
function app() {
  const $ = (s) => document.querySelector(s)
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c])
  const tier = (m) => (m === 'opus' ? 'deep' : m === 'haiku' ? 'mech' : 'run')

  // Contiguous runs of the same phase, so a repeated phase reads as a repeat
  // rather than collapsing into one card.
  const runsOf = (items, key) => {
    const out = []
    for (const it of items) {
      const last = out[out.length - 1]
      if (last && key(last.items[0]) === key(it) && key(it) !== null) last.items.push(it)
      else out.push({ key: key(it), items: [it] })
    }
    return out
  }

  let current = 0

  const node = (c, wi) => {
    const inst = /#(\d+)/.exec(c.label)
    const child = c.label.startsWith('workflow:') && c.label.slice(9)
    const target = child && DATA.findIndex((w) => w.name === child)
    return `<button class="node ${child ? 'child' : ''}" data-w="${wi}" data-n="${c.n}">
      <span class="n">${c.n}</span>
      <span class="lbl">${esc(c.label)}</span>
      ${inst ? `<span class="chip inst">#${inst[1]}</span>` : ''}
      <span class="grow"></span>
      ${child ? '<span class="chip inst">child workflow</span>' : `
        ${c.schema ? '<span class="chip schema">schema</span>' : '<span class="chip text">text</span>'}
        <span class="chip ${tier(c.model)}">${esc(c.model)}${c.effort ? ' &middot; ' + esc(c.effort) : ''}</span>`}
      <span class="chars">${(c.chars / 1000).toFixed(1)}k</span>
      ${child && target >= 0 ? `<span class="jump" data-go="${target}">open &rarr;</span>` : ''}
    </button>`
  }

  const renderWorkflow = (wi) => {
    const w = DATA[wi]
    const detail = Object.fromEntries(w.phases.map((p) => [p.title, p]))
    const body = runsOf(w.calls, (c) => c.phase)
      .map((blk) => {
        const p = detail[blk.key] || {}
        const inner = runsOf(blk.items, (c) => c.group)
          .map((run) =>
            run.key === null || run.items.length < 2
              ? run.items.map((c) => node(c, wi)).join('')
              : `<div class="fan"><div class="fan-tag">parallel &times;${run.items.length}</div>
                 <div class="fan-row">${run.items.map((c) => node(c, wi)).join('')}</div></div>`,
          )
          .join('')
        return `<section class="phase">
          <header><h3>${esc(blk.key || '(no phase)')}</h3>
            ${p.model ? `<span class="chip ${tier(p.model)}">${esc(p.model)}</span>` : ''}
            <span class="count">${blk.items.length}</span></header>
          ${p.detail ? `<p class="det">${esc(p.detail)}</p>` : ''}
          ${inner}
        </section>`
      })
      .join('<div class="link"></div>')

    $('#main').innerHTML = `
      <h1>${esc(w.name)}</h1>
      <p class="desc">${esc(w.description)}</p>
      <p class="when">${esc(w.whenToUse || '')}</p>
      <p class="args">dry run args: <code>${esc(JSON.stringify(w.args) || 'none')}</code>
        &nbsp;&middot;&nbsp; ${esc(w.name)}.js last edited ${esc(w.edited)}
</p>
      <div class="flow">${body}</div>
      <section class="tail">
        <h3>log output</h3>
        <ul>${w.logs.map((l) => `<li>${esc(l)}</li>`).join('') || '<li class="muted">none</li>'}</ul>
        <h3>returned</h3>
        <pre>${esc(JSON.stringify(w.result, null, 2))}</pre>
      </section>`
    $('#main').scrollTop = 0
    document.querySelectorAll('#side .wf').forEach((e, i) => e.classList.toggle('on', i === wi))
    current = wi
    filter()
  }

  const filter = () => {
    const q = $('#q').value.trim().toLowerCase()
    let hits = 0
    document.querySelectorAll('.node').forEach((n) => {
      const c = DATA[+n.dataset.w].calls.find((x) => x.n === +n.dataset.n)
      const hit = !q || (c.label + ' ' + c.prompt).toLowerCase().includes(q)
      n.classList.toggle('dim', !hit)
      if (q && hit) hits++
    })
    $('#hits').textContent = q ? `${hits} match` + (hits === 1 ? '' : 'es') : ''
  }

  document.body.insertAdjacentHTML(
    'afterbegin',
    `<aside id="side">
      <h2>workflows</h2>
      ${DATA.map(
        (w, i) => `<button class="wf" data-go="${i}">
          <b>${esc(w.name)}</b>
          <span>${w.calls.length} agents &middot; ${w.phases.length} phases</span></button>`,
      ).join('')}
      <div class="totals">
        built ${BUILT}<br>
        ${DATA.reduce((a, w) => a + w.calls.length, 0)} agents across
        ${DATA.length} workflows<br>
        ${(DATA.reduce((a, w) => a + w.calls.reduce((b, c) => b + c.chars, 0), 0) / 1000).toFixed(0)}k chars of prompt
      </div>
      <input id="q" placeholder="search labels and prompts">
      <div id="hits"></div>
    </aside>
    <main id="main"></main>
    <div id="drawer"><button id="close">close</button><div id="dbody"></div></div>`,
  )

  document.addEventListener('click', (e) => {
    const go = e.target.closest('[data-go]')
    if (go) {
      e.stopPropagation()
      return renderWorkflow(+go.dataset.go)
    }
    const n = e.target.closest('.node')
    if (n) {
      const c = DATA[+n.dataset.w].calls.find((x) => x.n === +n.dataset.n)
      $('#dbody').innerHTML = `<h2>${esc(c.label)}</h2>
        <p class="meta">${esc(c.phase)} &middot; call ${c.n} &middot; ${esc(c.model || '?')}${
        c.effort ? ' / ' + esc(c.effort) : ''
      } &middot; ${c.schema ? 'structured output' : 'free text'} &middot; ${c.chars} chars</p>
        <pre>${esc(c.prompt)}</pre>`
      $('#drawer').classList.add('open')
      $('#dbody').scrollTop = 0
      return
    }
    if (e.target.id === 'close') $('#drawer').classList.remove('open')
  })
  document.addEventListener('keydown', (e) => e.key === 'Escape' && $('#drawer').classList.remove('open'))
  $('#q').addEventListener('input', filter)
  renderWorkflow(0)
}

function render(data, built) {
  // JSON, not a JS literal: prompts are arbitrary text. Escaping < keeps a
  // "</script>" inside a prompt from ending the tag.
  const json = JSON.stringify(data).replace(/</g, '\u003c')
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Workflows</title>
<style>
:root{color-scheme:light dark;
  --bg:#f7f7f9;--panel:#fff;--ink:#1a1a20;--mute:#6b7280;--line:#e2e4e9;
  --side:#16161d;--side-ink:#d8d9e0;--accent:#7c6cf0;
  --deep:#b45309;--deep-bg:#fef3c7;--run:#047857;--run-bg:#d1fae5;--mech:#475569;--mech-bg:#e2e8f0}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f0f14;--panel:#17171f;--ink:#e6e6ec;--mute:#8b8fa0;--line:#26262f;
  --side:#0b0b10;--side-ink:#c9cad4;
  --deep:#fbbf24;--deep-bg:#3b2f0b;--run:#34d399;--run-bg:#0b3226;--mech:#94a3b8;--mech-bg:#1e293b}}
*{box-sizing:border-box}
body{margin:0;display:flex;height:100vh;overflow:hidden;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
#side{width:280px;flex:none;background:var(--side);color:var(--side-ink);padding:20px 16px;overflow:auto}
#side h2{margin:0 0 14px;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--mute)}
.wf{display:block;width:100%;text-align:left;background:none;border:0;border-radius:8px;padding:9px 10px;
  margin-bottom:2px;color:inherit;font:inherit;cursor:pointer}
.wf:hover{background:#ffffff14}
.wf.on{background:var(--accent);color:#fff}
.wf b{display:block;font-weight:600}
.wf span{font-size:12px;opacity:.65}
.totals{margin:18px 0;padding-top:14px;border-top:1px solid #ffffff1a;font-size:12px;color:var(--mute)}
#q{width:100%;padding:8px 10px;border-radius:8px;border:1px solid #ffffff1f;background:#ffffff0d;
  color:inherit;font:inherit}
#hits{font-size:12px;color:var(--mute);margin-top:6px}
#main{flex:1;overflow:auto;padding:32px 36px 90px;max-width:1000px}
h1{margin:0 0 6px;font-size:26px}
.desc{margin:0 0 12px;font-size:16px;color:var(--ink)}
.when,.args{margin:0 0 10px;font-size:13px;color:var(--mute);max-width:72ch}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
.phase{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.phase header{display:flex;align-items:center;gap:8px}
.phase h3{margin:0;font-size:15px}
.phase .count{margin-left:auto;font-size:12px;color:var(--mute)}
.det{margin:6px 0 12px;font-size:13px;color:var(--mute)}
.link{height:18px;margin-left:28px;border-left:2px solid var(--line)}
.node{display:flex;align-items:center;gap:8px;width:100%;text-align:left;margin-top:6px;
  padding:8px 10px;border:1px solid var(--line);border-radius:9px;background:var(--bg);
  color:inherit;font:inherit;cursor:pointer}
.node:hover{border-color:var(--accent)}
.node.dim{opacity:.22}
.node.child{border-style:dashed}
.node .n{font-size:11px;color:var(--mute);min-width:18px}
.node .lbl{font-weight:600;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12.5px}
.grow{flex:1}
.chars{font-size:11px;color:var(--mute);min-width:38px;text-align:right}
.chip{font-size:10.5px;padding:2px 7px;border-radius:99px;white-space:nowrap}
.chip.deep{background:var(--deep-bg);color:var(--deep)}
.chip.run{background:var(--run-bg);color:var(--run)}
.chip.mech,.chip.text,.chip.inst{background:var(--mech-bg);color:var(--mech)}
.chip.schema{background:var(--run-bg);color:var(--run)}
.jump{font-size:11px;color:var(--accent);font-weight:600}
.fan{margin-top:8px;padding:8px 10px;border:1px dashed var(--line);border-radius:10px}
.fan-tag{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}
.fan-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px}
.fan-row .node{margin-top:4px}
.tail{margin-top:34px;border-top:1px solid var(--line);padding-top:18px}
.tail h3{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--mute);margin:16px 0 6px}
.tail ul{margin:0;padding-left:18px;font-size:13px}
.muted{color:var(--mute)}
.stale{color:#f87171}
pre{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;
  overflow:auto;max-height:60vh;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
#drawer{display:none;position:fixed;top:0;right:0;width:min(660px,52vw);height:100%;
  background:var(--panel);border-left:1px solid var(--line);padding:20px 24px;overflow:auto}
#drawer.open{display:block}
#drawer h2{margin:0 0 4px;font-size:17px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
#drawer .meta{margin:0 0 14px;font-size:12px;color:var(--mute)}
#drawer pre{max-height:none}
#close{position:sticky;top:0;float:right;background:var(--bg);border:1px solid var(--line);
  border-radius:7px;padding:4px 10px;color:inherit;font:inherit;font-size:12px;cursor:pointer}
</style></head><body>
<script>const BUILT=${JSON.stringify(built)};const DATA=${json};(${app.toString()})()</script>
</body></html>
`
}
