"""Render evaluation results as Markdown, standalone HTML, and JUnit XML."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape
from xml.sax.saxutils import quoteattr

from .evals import EvalReport


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: EvalReport) -> str:
    gate = "PASS" if report.pass_rate >= 1.0 else "FAIL"
    lines = [
        f"# Behavioural evaluation scorecard: {report.suite}",
        "",
        f"**Gate: {gate}** · {report.passed}/{report.total} cases passed "
        f"({_pct(report.pass_rate)}) · mean score {report.mean_score:.3f} · "
        f"{report.duration_ms:.0f} ms",
        "",
        f"- Agent version: `{report.agent_version}`",
        f"- Dataset: `{report.dataset}`",
        f"- Generated: `{report.generated_at}`",
        f"- Reproduce: `PYTHONPATH=src python -m iai evals run --suite data/evals/core.jsonl`",
        "",
        "## Results by capability",
        "",
        "| Tag | Cases | Passed | Pass rate | Mean score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for tag, stats in report.by_tag().items():
        lines.append(
            f"| `{tag}` | {stats['total']} | {stats['passed']} | "
            f"{_pct(stats['pass_rate'])} | {stats['mean_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Grader agreement",
            "",
            "| Grader | Checks | Passed | Pass rate | Mean score |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for kind, stats in report.criteria_summary().items():
        lines.append(
            f"| `{kind}` | {stats['total']} | {stats['passed']} | "
            f"{_pct(stats['pass_rate'])} | {stats['mean_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Case scorecard",
            "",
            "| Case | Status | Score | Result | Failed criteria |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for result in report.results:
        outcome = "PASS" if result.passed else "FAIL"
        failed = ", ".join(f"`{kind}`" for kind in result.failed_criteria) or "—"
        lines.append(
            f"| `{result.case_id}` {result.name} | `{result.status}` | "
            f"{result.score:.3f} | {outcome} | {failed} |"
        )
    if report.failures:
        lines.extend(["", "## Failure detail", ""])
        for result in report.failures:
            lines.append(f"### `{result.case_id}` — {result.name}")
            lines.append("")
            if result.error:
                lines.append(f"- Error: `{result.error}`")
            answer = result.answer or {}
            if answer:
                lines.append(f"- Summary: {answer.get('summary', '')}")
                if answer.get("diagnosis"):
                    lines.append(f"- Diagnosis: {answer.get('diagnosis')}")
                if answer.get("escalation_reason"):
                    lines.append(f"- Escalation: {answer.get('escalation_reason')}")
            lines.append("- Criteria:")
            for criterion in result.criteria:
                mark = "pass" if criterion.passed else "FAIL"
                lines.append(
                    f"  - `{criterion.kind}` ({mark}, {criterion.score:.2f}): {criterion.detail}"
                )
            lines.append("")
    lines.extend(
        [
            "",
            "---",
            "",
            "Scores are produced by deterministic graders over the answer, tool trace,",
            "evidence ledger, and guardrail verdict. No LLM judge is used.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(report: EvalReport) -> str:
    gate = "PASS" if report.pass_rate >= 1.0 else "FAIL"
    gate_class = "pass" if gate == "PASS" else "fail"
    rows = []
    for result in report.results:
        outcome = "PASS" if result.passed else "FAIL"
        css = "pass" if result.passed else "fail"
        failed = ", ".join(result.failed_criteria) or "—"
        answer = result.answer or {}
        criterion_rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(item.kind)}</code></td>"
            f"<td class=\"{'pass' if item.passed else 'fail'}\">{'pass' if item.passed else 'FAIL'}</td>"
            f"<td>{item.score:.2f}</td>"
            f"<td>{html.escape(item.detail)}</td>"
            "</tr>"
            for item in result.criteria
        )
        tool_names = ", ".join(
            call.get("name", "?") for call in answer.get("tool_calls", [])
        )
        rows.append(
            f"""
            <tr class="case-row" data-status="{css}">
              <td><code>{html.escape(result.case_id)}</code><br><span class="muted">{html.escape(result.name)}</span></td>
              <td><span class="pill {css}">{outcome}</span></td>
              <td class="score">{result.score:.3f}</td>
              <td><code>{html.escape(result.status)}</code><br><span class="muted">{result.duration_ms:.0f} ms</span></td>
              <td>{html.escape(failed)}</td>
              <td>
                <details>
                  <summary>Inspect</summary>
                  <p><strong>Summary.</strong> {html.escape(str(answer.get('summary', '')))}</p>
                  <p><strong>Diagnosis.</strong> {html.escape(str(answer.get('diagnosis', '')))}</p>
                  <p><strong>Escalation.</strong> {html.escape(str(answer.get('escalation_reason') or '—'))}</p>
                  <p><strong>Tools.</strong> <code>{html.escape(tool_names or '—')}</code></p>
                  <table><thead><tr><th>Grader</th><th>Result</th><th>Score</th><th>Detail</th></tr></thead>
                  <tbody>{criterion_rows}</tbody></table>
                </details>
              </td>
            </tr>
            """
        )
    tag_rows = "".join(
        f"<tr><td><code>{html.escape(tag)}</code></td><td>{stats['passed']}/{stats['total']}</td>"
        f"<td>{_pct(stats['pass_rate'])}</td><td>{stats['mean_score']:.3f}</td></tr>"
        for tag, stats in report.by_tag().items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Industrial Agent Intelligence — eval scorecard</title>
<style>
  :root {{
    --bg: #0b1220; --panel: #111b2e; --panel-2: #16233b; --ink: #e8eefc;
    --muted: #93a4c3; --line: #243552; --accent: #5eead4; --pass: #34d399;
    --fail: #fb7185; --warn: #fbbf24;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: radial-gradient(1200px 600px at 20% -10%, #16324f 0%, var(--bg) 55%);
    color: var(--ink); font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  header, main {{ max-width: 1180px; margin: 0 auto; padding: 28px 22px; }}
  header {{ padding-bottom: 8px; }}
  h1 {{ font-size: clamp(1.6rem, 3vw, 2.4rem); margin: 0 0 6px; letter-spacing: -0.02em; }}
  .sub {{ color: var(--muted); margin: 0; }}
  .badge {{ display: inline-block; padding: 5px 12px; border-radius: 999px; font-weight: 700;
    letter-spacing: .06em; font-size: .78rem; margin-right: 8px; }}
  .badge.pass {{ background: rgba(52,211,153,.14); color: var(--pass); border: 1px solid rgba(52,211,153,.35); }}
  .badge.fail {{ background: rgba(251,113,133,.14); color: var(--fail); border: 1px solid rgba(251,113,133,.35); }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 22px 0; }}
  .card {{ background: linear-gradient(180deg, var(--panel), var(--panel-2)); border: 1px solid var(--line);
    border-radius: 14px; padding: 16px; }}
  .card .k {{ color: var(--muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .09em; }}
  .card .v {{ font-size: 1.7rem; font-weight: 700; margin-top: 4px; }}
  .card .v.pass {{ color: var(--pass); }} .card .v.fail {{ color: var(--fail); }}
  .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px; margin: 18px 0; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 12px; letter-spacing: .01em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th, td {{ text-align: left; padding: 8px 9px; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; font-size: .74rem; text-transform: uppercase; letter-spacing: .07em; }}
  tr:last-child td {{ border-bottom: none; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .86em; }}
  .muted {{ color: var(--muted); }}
  .pill {{ padding: 3px 9px; border-radius: 999px; font-size: .72rem; font-weight: 700; }}
  .pill.pass {{ background: rgba(52,211,153,.14); color: var(--pass); }}
  .pill.fail {{ background: rgba(251,113,133,.14); color: var(--fail); }}
  .score {{ font-variant-numeric: tabular-nums; font-weight: 700; }}
  details summary {{ cursor: pointer; color: var(--accent); }}
  details p {{ margin: 8px 0; }}
  .foot {{ color: var(--muted); font-size: .8rem; margin-top: 26px; }}
  @media (max-width: 760px) {{ table {{ font-size: .8rem; }} th, td {{ padding: 6px; }} }}
</style>
</head>
<body>
<header>
  <span class="badge {gate_class}">GATE {gate}</span>
  <span class="badge">v{html.escape(report.agent_version)}</span>
  <h1>Industrial agent behavioural scorecard</h1>
  <p class="sub">Suite <code>{html.escape(report.suite)}</code> · dataset <code>{html.escape(report.dataset)}</code>
     · generated <code>{html.escape(report.generated_at)}</code></p>
</header>
<main>
  <section class="cards">
    <div class="card"><div class="k">Cases passed</div><div class="v {'pass' if report.pass_rate == 1 else 'fail'}">{report.passed}/{report.total}</div></div>
    <div class="card"><div class="k">Pass rate</div><div class="v {'pass' if report.pass_rate == 1 else 'fail'}">{_pct(report.pass_rate)}</div></div>
    <div class="card"><div class="k">Mean score</div><div class="v">{report.mean_score:.3f}</div></div>
    <div class="card"><div class="k">Runtime</div><div class="v">{report.duration_ms:.0f}<span class="muted" style="font-size:.9rem"> ms</span></div></div>
  </section>
  <section class="panel">
    <h2>Capability coverage</h2>
    <table><thead><tr><th>Tag</th><th>Passed</th><th>Pass rate</th><th>Mean score</th></tr></thead><tbody>{tag_rows}</tbody></table>
  </section>
  <section class="panel">
    <h2>Case scorecard</h2>
    <table><thead><tr><th>Case</th><th>Gate</th><th>Score</th><th>Status</th><th>Failed criteria</th><th></th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table>
  </section>
  <p class="foot">Deterministic graders over tool trace, evidence ledger, guardrail verdict, and answer text.
  No model-as-judge. Generated by <code>python -m iai evals run</code>.</p>
</main>
</body>
</html>
"""


def render_junit(report: EvalReport) -> str:
    failures = len(report.failures)
    suite_attrs = (
        f"name={quoteattr(report.suite)} tests=\"{report.total}\" "
        f"failures=\"{failures}\" errors=\"0\" time=\"{report.duration_ms / 1000:.3f}\""
    )
    cases = []
    for result in report.results:
        case_time = result.duration_ms / 1000
        if result.passed:
            cases.append(
                f'<testcase classname="iai.evals" name={quoteattr(result.case_id)} '
                f'time="{case_time:.3f}"/>'
            )
            continue
        detail = result.error or "; ".join(
            f"{item.kind}: {item.detail}" for item in result.criteria if not item.passed
        )
        cases.append(
            f'<testcase classname="iai.evals" name={quoteattr(result.case_id)} time="{case_time:.3f}">'
            f'<failure message={quoteattr(result.name)}>{xml_escape(detail)}</failure>'
            "</testcase>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite {suite_attrs}>\n' + "\n".join(cases) + "\n</testsuite>\n"
    )


def write_report(report: EvalReport, directory: str | Path, *, stem: str = "latest") -> dict[str, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": target / f"{stem}.json",
        "markdown": target / f"{stem}.md",
        "html": target / f"{stem}.html",
        "junit": target / f"{stem}-junit.xml",
    }
    paths["json"].write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    paths["html"].write_text(render_html(report), encoding="utf-8")
    paths["junit"].write_text(render_junit(report), encoding="utf-8")
    return paths


def load_report(path: str | Path) -> EvalReport:
    return EvalReport.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def report_summary(report: EvalReport) -> dict[str, Any]:
    return {
        "suite": report.suite,
        "pass_rate": report.pass_rate,
        "passed": report.passed,
        "total": report.total,
        "mean_score": report.mean_score,
        "failures": [result.case_id for result in report.failures],
    }
