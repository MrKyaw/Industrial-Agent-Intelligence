"""Command-line interface for demos, evaluation gates, and the read-only API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .alarms import AlarmError
from .api import ApiConfig, serve
from .app import AppConfig, Application, default_data_dir
from .evals import EvalError, EvalRunner, compare_reports, load_cases
from .feedback import FeedbackError, FeedbackStore
from .ontology import OntologyError
from .prompts import PromptError
from .report import load_report, report_summary, write_report
from .retrieval import RetrievalError
from .telemetry import TelemetryError
from .tools import PolicyError, ToolError

DEFAULT_QUESTION = (
    "Why is Chiller 1 tripping high condenser pressure alarms and what should we check?"
)
DEFAULT_SITE = "site.dc-lon-01"
DEFAULT_SUITE = "data/evals/core.jsonl"
DEFAULT_REPORT_DIR = "reports/evals"
DEFAULT_FEEDBACK_DB = "reports/feedback/feedback.db"


def _application(args: argparse.Namespace) -> Application:
    data_dir = Path(args.data_dir) if getattr(args, "data_dir", None) else default_data_dir()
    config = AppConfig(allow_physical_control=bool(getattr(args, "allow_writes", False)))
    return Application.load(data_dir, config=config)


def _print_answer(answer: Any) -> None:
    print(f"status:       {answer.status}")
    print(f"intent:       {answer.intent}")
    print(f"confidence:   {answer.confidence:.2f}")
    print(f"escalated:    {answer.escalated}")
    print(f"tools:        {len(answer.tool_calls)}")
    print(f"guardrails:   {'pass' if answer.guardrails.get('passed') else 'BLOCKED'}")
    print(f"duration:     {answer.duration_ms:.1f} ms")
    print()
    print(answer.summary)
    if answer.diagnosis:
        print()
        print(answer.diagnosis)
    if answer.recommendations:
        print()
        print("Next checks:")
        for index, recommendation in enumerate(answer.recommendations, start=1):
            print(f"  {index}. {recommendation}")
    if answer.escalation_reason:
        print()
        print(f"Escalation: {answer.escalation_reason}")
    if answer.disclosure:
        print()
        print(f"Disclosure: {answer.disclosure}")
    if answer.citations:
        print()
        print(f"Evidence: {', '.join(answer.citations)}")


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"industrial-agent-intelligence {__version__}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    app = _application(args)
    answer = app.ask(args.question, site_id=args.site, role=args.role)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
    else:
        _print_answer(answer)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    app = _application(args)
    answer = app.ask(args.question, site_id=args.site, role=args.role)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
    else:
        _print_answer(answer)
    return 0 if answer.status != "error" else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    app = _application(args)
    report = app.validate()
    suite_path = Path(args.suite)
    if suite_path.exists():
        cases = load_cases(suite_path)
        report["eval_cases"] = len(cases)
        report["eval_ids"] = [case.id for case in cases]
    report["prompts"] = app.prompts.ids()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"ontology:   {'ok' if report['ontology']['ok'] else 'FAILED'}")
        print(f"series:     {report['telemetry_series']}")
        print(f"knowledge:  {report['knowledge_documents']}")
        print(f"alarms:     {report['alarms']}")
        print(f"tools:      {report['tools']}")
        print(f"evals:      {report.get('eval_cases', 'not loaded')}")
        print(f"prompts:    {', '.join(report['prompts'])}")
        for error in report["ontology"]["errors"]:
            print(f"  error: {error}", file=sys.stderr)
    return 0 if report["ontology"]["ok"] else 1


def _cmd_evals_run(args: argparse.Namespace) -> int:
    app = _application(args)
    if app.runtime is None:
        raise RuntimeError("runtime is not ready")
    suite_path = Path(args.suite)
    cases = load_cases(suite_path)
    runner = EvalRunner(app.runtime)
    report = runner.run(cases, suite=args.suite_name or suite_path.stem, dataset=str(app.manifest.get("dataset", "unknown")))
    paths = write_report(report, args.out, stem=args.stem)
    summary = report_summary(report)
    print(
        f"{report.suite}: {report.passed}/{report.total} passed "
        f"({report.pass_rate * 100:.1f}%), mean score {report.mean_score:.3f}, "
        f"{report.duration_ms:.0f} ms"
    )
    for result in report.failures:
        print(f"  FAIL {result.case_id}: {', '.join(result.failed_criteria) or result.error}", file=sys.stderr)
    print(f"markdown: {paths['markdown']}")
    print(f"html:     {paths['html']}")
    print(f"json:     {paths['json']}")
    print(f"junit:    {paths['junit']}")
    gate = report.gate(args.min_pass_rate)
    if not gate:
        print(
            f"gate failed: pass rate {report.pass_rate:.3f} < {args.min_pass_rate:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_evals_compare(args: argparse.Namespace) -> int:
    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    comparison = compare_reports(baseline, candidate, tolerance=args.tolerance)
    print(json.dumps(comparison.to_dict(), indent=2))
    return 0 if comparison.passed else 1


def _cmd_report(args: argparse.Namespace) -> int:
    report = load_report(args.report)
    paths = write_report(report, args.out, stem=args.stem)
    print(f"wrote {paths['markdown']}")
    print(f"wrote {paths['html']}")
    print(f"wrote {paths['junit']}")
    return 0


def _feedback_store(args: argparse.Namespace) -> FeedbackStore:
    return FeedbackStore(Path(args.db))


def _cmd_feedback(args: argparse.Namespace) -> int:
    store = _feedback_store(args)
    if args.feedback_command == "add":
        record = store.add(
            request_id=args.request_id,
            rating=args.rating,
            agent_version=args.agent_version,
            label=args.label,
            comment=args.comment,
            site_id=args.site,
            slice_name=args.slice,
        )
        print(json.dumps(record.to_dict(), indent=2))
        return 0
    if args.feedback_command == "list":
        records = store.list(limit=args.limit, site_id=args.site or None, label=args.label or None)
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return 0
    if args.feedback_command == "label":
        record = store.label(args.id, args.label)
        print(json.dumps(record.to_dict(), indent=2))
        return 0
    if args.feedback_command == "pending":
        records = store.pending(limit=args.limit)
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return 0
    if args.feedback_command == "summary":
        print(json.dumps(store.summary(), indent=2))
        return 0
    if args.feedback_command == "drift":
        report = store.drift(recent_n=args.recent, baseline_n=args.baseline)
        print(json.dumps(report.to_dict(), indent=2))
        return 0
    if args.feedback_command == "export":
        count = store.export_jsonl(args.output)
        print(f"exported {count} records to {args.output}")
        return 0
    raise FeedbackError(f"unknown feedback command: {args.feedback_command}")


def _cmd_serve(args: argparse.Namespace) -> int:
    app = _application(args)
    token = args.token or os.environ.get("IAI_API_TOKEN") or None
    config = ApiConfig(
        host=args.host,
        port=args.port,
        token=token,
        rate_limit_per_minute=args.rate_limit,
        verbose=args.verbose,
    )
    print(f"serving {app.status()['dataset']} at http://{args.host}:{args.port} (Ctrl-C to stop)")
    serve(app, config)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iai",
        description="Industrial agent reasoning, tool policy, guardrails, and behavioural evals.",
    )
    parser.add_argument("--data-dir", help="dataset directory (default: bundled data/)")
    parser.add_argument("--allow-writes", action="store_true", help="enable physical-control tools")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version = subparsers.add_parser("version", help="print the package version")
    version.set_defaults(func=_cmd_version)

    demo = subparsers.add_parser("demo", help="run the reference triage scenario")
    demo.add_argument("--question", default=DEFAULT_QUESTION)
    demo.add_argument("--site", default=DEFAULT_SITE)
    demo.add_argument("--role", default="operator")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=_cmd_demo)

    ask = subparsers.add_parser("ask", help="ask the agent a question")
    ask.add_argument("question")
    ask.add_argument("--site", default=DEFAULT_SITE)
    ask.add_argument("--role", default="operator")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=_cmd_ask)

    validate = subparsers.add_parser("validate", help="validate data, prompts, and eval suite")
    validate.add_argument("--suite", default=DEFAULT_SUITE)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=_cmd_validate)

    evals = subparsers.add_parser("evals", help="run or compare behavioural evaluations")
    eval_sub = evals.add_subparsers(dest="evals_command", required=True)

    run = eval_sub.add_parser("run", help="run an eval suite and write scorecards")
    run.add_argument("--suite", default=DEFAULT_SUITE)
    run.add_argument("--suite-name")
    run.add_argument("--out", default=DEFAULT_REPORT_DIR)
    run.add_argument("--stem", default="latest")
    run.add_argument("--min-pass-rate", type=float, default=1.0)
    run.set_defaults(func=_cmd_evals_run)

    compare = eval_sub.add_parser("compare", help="fail on regressions against a baseline")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--tolerance", type=float, default=0.02)
    compare.set_defaults(func=_cmd_evals_compare)

    report = subparsers.add_parser("report", help="re-render a stored JSON report")
    report.add_argument("report")
    report.add_argument("--out", default=DEFAULT_REPORT_DIR)
    report.add_argument("--stem", default="latest")
    report.set_defaults(func=_cmd_report)

    serve_parser = subparsers.add_parser("serve", help="serve the read-only API and dashboard")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--token", default=None)
    serve_parser.add_argument("--rate-limit", type=int, default=120)
    serve_parser.add_argument("--verbose", action="store_true")
    serve_parser.set_defaults(func=_cmd_serve)

    feedback = subparsers.add_parser("feedback", help="record and analyse operator feedback")
    feedback.add_argument("--db", default=DEFAULT_FEEDBACK_DB)
    feedback_sub = feedback.add_subparsers(dest="feedback_command", required=True)

    add = feedback_sub.add_parser("add")
    add.add_argument("--request-id", required=True)
    add.add_argument("--rating", type=int, required=True)
    add.add_argument("--agent-version", default=__version__)
    add.add_argument("--label", default="")
    add.add_argument("--comment", default="")
    add.add_argument("--site", default="")
    add.add_argument("--slice", default="")
    add.set_defaults(func=_cmd_feedback)

    listing = feedback_sub.add_parser("list")
    listing.add_argument("--limit", type=int, default=50)
    listing.add_argument("--site")
    listing.add_argument("--label")
    listing.set_defaults(func=_cmd_feedback)

    label = feedback_sub.add_parser("label")
    label.add_argument("id", type=int)
    label.add_argument("--label", required=True)
    label.set_defaults(func=_cmd_feedback)

    pending = feedback_sub.add_parser("pending")
    pending.add_argument("--limit", type=int, default=50)
    pending.set_defaults(func=_cmd_feedback)

    summary = feedback_sub.add_parser("summary")
    summary.set_defaults(func=_cmd_feedback)

    drift = feedback_sub.add_parser("drift")
    drift.add_argument("--recent", type=int, default=50)
    drift.add_argument("--baseline", type=int, default=200)
    drift.set_defaults(func=_cmd_feedback)

    export = feedback_sub.add_parser("export")
    export.add_argument("output")
    export.set_defaults(func=_cmd_feedback)

    for _sub in feedback_sub.choices.values():
        _sub.add_argument("--db", default=DEFAULT_FEEDBACK_DB)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (
        AlarmError,
        EvalError,
        FeedbackError,
        FileNotFoundError,
        OntologyError,
        PolicyError,
        PromptError,
        RetrievalError,
        TelemetryError,
        ToolError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
