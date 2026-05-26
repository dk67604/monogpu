#!/usr/bin/env bash
# Generate a self-contained HTML report from vllm-bench-serve result JSONs.
# Pure shell: embeds the JSON files into the page and lets Chart.js (CDN) parse
# them client-side — no jq, no Python. Open the result in any browser.
#
#   bench/chart.sh [out/report.html]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/out}"
HTML="${1:-${OUT_DIR}/report.html}"

shopt -s nullglob
files=("$OUT_DIR"/*.json)
if [ "${#files[@]}" -eq 0 ]; then
  echo "[chart] no result JSONs in $OUT_DIR — run bench/run_bench.sh or bench/sweep.sh first" >&2
  exit 1
fi

{
  cat <<'HTML_HEAD'
<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>monogpu — single-GPU multi-model benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{font-family:system-ui,sans-serif;margin:2rem;background:#0f1115;color:#e6e6e6}
 h1{margin:0 0 .2rem} .sub{color:#9aa;margin-bottom:1.5rem}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
 .card{background:#171a21;border:1px solid #242a35;border-radius:10px;padding:1rem}
 canvas{max-height:340px}
 code{background:#222;padding:.1rem .3rem;border-radius:4px}
</style></head><body>
<h1>monogpu — single-GPU multi-model benchmark</h1>
<div class="sub">RTX 5080 (sm_120) · SMG gateway → vLLM workers · charts from <code>vllm bench serve</code> results</div>
<div class="grid">
 <div class="card"><h3>Output throughput (tok/s)</h3><canvas id="c_tput"></canvas></div>
 <div class="card"><h3>P99 TTFT (ms)</h3><canvas id="c_ttft"></canvas></div>
 <div class="card"><h3>P99 TPOT (ms)</h3><canvas id="c_tpot"></canvas></div>
 <div class="card"><h3>Request goodput (req/s, if SLO set)</h3><canvas id="c_good"></canvas></div>
</div>
<script>
const RESULTS = [
HTML_HEAD

  first=1
  for f in "${files[@]}"; do
    [ "$first" -eq 1 ] || printf ',\n'
    cat "$f"
    first=0
  done

  cat <<'HTML_TAIL'
];
// Normalize request_rate: numeric, or Infinity for "inf".
function rr(r){ const v=r.request_rate; const n=Number(v);
  return (v==="inf"||!isFinite(n))?Infinity:n; }
function rrLabel(v){ return v===Infinity?"inf":String(v); }

// x-axis = sorted distinct request rates; one line per model_id.
const rates=[...new Set(RESULTS.map(rr))].sort((a,b)=>a-b);
const labels=rates.map(rrLabel);
const models=[...new Set(RESULTS.map(r=>r.model_id))].sort();
const palette=["#5eb0ef","#f59e0b","#34d399","#f472b6","#a78bfa","#f87171"];

function dataset(metric){
  return models.map((m,i)=>{
    const pts=rates.map(rate=>{
      const hit=RESULTS.find(r=>r.model_id===m && rr(r)===rate);
      return hit?hit[metric]:null;
    });
    return {label:m.split("/").pop(), data:pts, borderColor:palette[i%palette.length],
            backgroundColor:palette[i%palette.length], spanGaps:true, tension:.2};
  });
}
function mk(id,metric){
  new Chart(document.getElementById(id),{type:"line",
    data:{labels,datasets:dataset(metric)},
    options:{responsive:true,plugins:{legend:{labels:{color:"#ccc"}}},
      scales:{x:{title:{display:true,text:"request rate (req/s)",color:"#9aa"},ticks:{color:"#9aa"}},
              y:{beginAtZero:true,ticks:{color:"#9aa"}}}}});
}
mk("c_tput","output_throughput");
mk("c_ttft","p99_ttft_ms");
mk("c_tpot","p99_tpot_ms");
mk("c_good","request_goodput");
</script></body></html>
HTML_TAIL
} > "$HTML"

echo "[chart] wrote $HTML  ($(wc -c < "$HTML") bytes, ${#files[@]} result files)"
echo "[chart] open it: file://$HTML"
