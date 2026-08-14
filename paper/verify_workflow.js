export const meta = {
  name: 'verify-references',
  description: 'Ground every bib claim against its local PDF excerpt (direction+magnitude) and sweep for missed concurrent work',
  phases: [
    { title: 'Ground', detail: 'per-reference: does the PDF excerpt support the claim direction+number' },
    { title: 'Discover', detail: 'mechanism-name + citation-graph + concurrent-window sweep for un-cited related work' },
  ],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const DIR = A.dir
const KEYS = A.keys
const CITED = `${DIR}/refpdfs/cited.json`
// load-bearing concurrent-work entries get the stronger model
const HEAVY = new Set(['kayyam2026sticky', 'eldr2026', 'cachecond2024'])

const GROUND_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['key', 'status', 'direction_ok', 'quote', 'page', 'note'],
  properties: {
    key: { type: 'string' },
    status: { enum: ['CONFIRMED', 'PARTIAL', 'MISMATCH', 'UNSUPPORTED'] },
    direction_ok: { type: 'boolean', description: 'does the excerpt support the claim sign/direction (improves vs degrades, faster vs slower)' },
    quote: { type: 'string', description: 'exact sentence(s) from the excerpt that support or contradict the claim' },
    page: { type: 'string' },
    note: { type: 'string', description: 'if not CONFIRMED, what specifically differs' },
  },
}

const DISCOVER_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['angle', 'candidates'],
  properties: {
    angle: { type: 'string' },
    candidates: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['title', 'arxiv_id', 'year', 'mechanism', 'already_cited', 'threat_level', 'why_relevant'],
        properties: {
          title: { type: 'string' },
          arxiv_id: { type: 'string', description: 'arXiv id if known, else empty' },
          year: { type: 'string' },
          mechanism: { type: 'string', description: 'what it does, in a few words' },
          already_cited: { type: 'boolean', description: 'is it already in cited.json' },
          threat_level: { enum: ['HIGH', 'MED', 'LOW'], description: 'HIGH = pre-empts a core claim of our paper' },
          why_relevant: { type: 'string' },
        },
      },
    },
  },
}

const THESIS =
  'Our paper: training-time router locality loss (penalise expert switches) is NOT free on a 137M ' +
  'multi-domain MoE (+2-3% PPL), BUT training-time locality STACKS with training-free cache-aware ' +
  'rerouting (tolerance tau at inference) to reach ~80% expert-cache-miss reduction at +2-3% PPL. ' +
  'Concurrent work we already cite: StickyMoE (routing-consistency loss), cache-conditional experts ' +
  '(training-free tolerance routing), EdgeMoE. We must find any OTHER same-mechanism work we missed.'

// ---- Phase 1: Ground each claim against its local PDF excerpt ----
phase('Ground')
const grounded = await parallel(KEYS.map((key) => async () => {
  const heavy = HEAVY.has(key)
  const r = await agent(
    `Read the file ${DIR}/refpdfs/ctx/${key}.txt . It contains a bib CLAIM (verified_finding) and ` +
    `page-tagged EXCERPTS from the paper's own PDF. Judge ONLY from the excerpts (no outside knowledge). ` +
    `Decide whether the excerpts support the claim's DIRECTION (e.g. perplexity IMPROVES vs DEGRADES, ` +
    `speedup vs slowdown) AND its numbers. A number appearing with the opposite sign/meaning is a MISMATCH, ` +
    `not a confirmation. If the excerpt lacks the needed sentence, status=UNSUPPORTED. Quote the exact ` +
    `supporting/contradicting sentence and its ~page. Set key="${key}".`,
    { label: `ground:${key}`, phase: 'Ground', schema: GROUND_SCHEMA,
      model: heavy ? 'opus' : 'sonnet', effort: heavy ? 'medium' : 'low' }
  )
  return r
}))

// ---- Phase 2: Discovery sweep for missed concurrent / same-mechanism work ----
phase('Discover')
const ANGLES = [
  { label: 'mechanism-names', prompt:
    `Search the web (use WebSearch/WebFetch; load them via ToolSearch if needed) for MoE papers whose ` +
    `MECHANISM matches ours, by mechanism name not title: "router temporal locality loss", ` +
    `"expert reuse / switch penalty", "routing consistency loss", "sticky expert routing", ` +
    `"cache-aware MoE routing", "training-free expert rerouting tolerance". ` },
  { label: 'citation-graph', prompt:
    `Use the web to find papers that CITE or are cited alongside cache-conditional experts ` +
    `(arXiv:2412.00099, Skliar et al.) and EdgeMoE (arXiv:2308.14352). We want same-mechanism neighbours ` +
    `in that citation graph (Semantic Scholar / Google Scholar / arXiv listings). ` },
  { label: 'concurrent-window', prompt:
    `Search for MoE expert-cache-locality / expert-reuse / edge-MoE routing papers from the CONCURRENT ` +
    `window late-2025 through 2026 (our submission is mid-2026). List any that train or reroute for ` +
    `expert locality/cache-hit. ` },
  { label: 'completeness-critic', prompt:
    `Given our thesis, what obvious same-mechanism or directly-competing work is MISSING from our ` +
    `citation list? Think adversarially: what would a hostile reviewer say we failed to cite? ` },
]
const discovered = await parallel(ANGLES.map((a) => async () => {
  return await agent(
    `${THESIS}\n\nFirst Read ${CITED} (our 56 cited papers: key, arxiv id, title) so you can set ` +
    `already_cited correctly. ${a.prompt}\nReturn up to 8 candidates, most-threatening first. For each, ` +
    `set already_cited by checking cited.json, and threat_level=HIGH only if it pre-empts a CORE claim ` +
    `(training-time locality loss for cache, OR training-free tolerance rerouting, OR their stacking). ` +
    `angle="${a.label}".`,
    { label: `discover:${a.label}`, phase: 'Discover', schema: DISCOVER_SCHEMA,
      model: 'sonnet', effort: 'medium' }
  )
}))

return {
  grounded: grounded.filter(Boolean),
  discovered: discovered.filter(Boolean),
}
