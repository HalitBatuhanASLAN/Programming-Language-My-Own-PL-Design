#!/usr/bin/env python3
"""
ayar_test_suite.py
==================
Auto-discovering test runner for the Ayar DSL implementation.

Scans the script's own directory for .ayar files, infers the expected
outcome from each file's name prefix, runs the full pipeline, and reports
PASS / FAIL / SKIP with per-test timing.

FILE NAMING CONVENTION
----------------------
  2l*   → Lexer tests  — expected to run cleanly (all l-files pass after fixes).
  2p*   → Parser tests — files whose name contains "int_ratio" / "error" / "bad"
                         are expected to raise ParseError; all others run cleanly.
  2tc*  → Type-checker tests:
              tc1 – tc4  → expect ≥ 1 AyarTypeError
              tc5 (valid)→ expect clean execution
  2i*   → Interpreter tests — expected to run cleanly and produce output.
  sample_errors.ayar      → expected to fail (LayoutError in that file).
  sample_5.ayar
  sample_minimal.ayar     → expected to produce AyarTypeError(s).
  All other sample_*       → expected to run cleanly.

HOW TO RUN
----------
  1. Put this script in the same folder as:
       ayar_lexer.py  ayar_parser.py  ayar_typechecker.py  ayar_interpreter.py
       2l1_block_comment.ayar  2l2_string_escapes.ayar  …  (all your .ayar files)

  2. Run:
       python3 ayar_test_suite.py

  Optional flags:
    --verbose           Print full actual output / error detail for every test
    --stop-first        Stop on the first failure
    --category CAT      Only run one category:
                        lexer | parser | typechecker | interpreter | sample

REQUIREMENTS
------------
  Python 3.7+  —  no third-party packages needed.
"""

import argparse
import glob
import io
import os
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_cli = argparse.ArgumentParser(
    description="Ayar DSL — auto-discovering test runner",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_cli.add_argument("--verbose",    action="store_true",
                  help="Print full actual output / error detail for every test")
_cli.add_argument("--stop-first", action="store_true",
                  help="Stop on the first failing test")
_cli.add_argument("--category",   default="", metavar="CAT",
                  help=("Run only: lexer | parser | typechecker "
                        "| interpreter | sample"))
ARGS = _cli.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_COLOUR = sys.stdout.isatty()

def _e(code, text):  return f"\033[{code}m{text}\033[0m" if _COLOUR else text
def green(t):        return _e("92", t)
def red(t):          return _e("91", t)
def yellow(t):       return _e("93", t)
def cyan(t):         return _e("96", t)
def bold(t):         return _e("1",  t)
def dim(t):          return _e("2",  t)

# ─────────────────────────────────────────────────────────────────────────────
# LOCATE IMPLEMENTATION FILES
# ─────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_IMPL_MODULES = ("ayar_lexer", "ayar_parser", "ayar_typechecker", "ayar_interpreter")

def _check_impl():
    missing = [m + ".py" for m in _IMPL_MODULES
               if not os.path.exists(os.path.join(HERE, m + ".py"))]
    if missing:
        print(red(f"\n  ERROR: Missing implementation file(s): {missing}"))
        print(f"  Expected in: {HERE}")
        sys.exit(1)

_check_impl()

from ayar_lexer       import Lexer, LexicalError                    # noqa: E402
from ayar_parser      import Parser, ParseError, LayoutError, validate_layout  # noqa: E402
from ayar_typechecker import TypeChecker, AyarTypeError             # noqa: E402
from ayar_interpreter import run_source                             # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lex(src: str):
    return Lexer(src).tokenize()

def _parse(src: str):
    tokens = _lex(src)
    validate_layout(tokens)
    return Parser(tokens).parse(), tokens

def _typecheck(src: str):
    ast, tokens = _parse(src)
    return TypeChecker(ast, tokens).check()

def _run(src: str):
    """Full pipeline. Returns (stdout_str, exception|None). Never raises."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    exc = None
    try:
        run_source(src)
    except Exception as e:
        exc = e
    finally:
        sys.stdout = old
    return buf.getvalue(), exc

# ─────────────────────────────────────────────────────────────────────────────
# FILE CLASSIFIER
#
# Returns (category, expectation) for a given filename.
#
# category    : lexer | parser | typechecker | interpreter | sample | unknown
# expectation : clean | parse_error | type_errors | skip
# ─────────────────────────────────────────────────────────────────────────────

def _classify(filename: str):
    b = filename.lower()

    if b.startswith("2l"):
        # All lexer files are expected to run cleanly after the block-comment fix.
        return "lexer", "clean"

    if b.startswith("2p"):
        # p3 deliberately uses integer ratios → ParseError.
        if "int_ratio" in b or "error" in b or "bad" in b:
            return "parser", "parse_error"
        return "parser", "clean"

    if b.startswith("2tc"):
        # tc5 is the "valid" baseline — zero errors expected.
        # tc1-tc4 all expect at least one type error.
        if "valid" in b or b.startswith("2tc5"):
            return "typechecker", "clean"
        return "typechecker", "type_errors"

    if b.startswith("2i"):
        return "interpreter", "clean"

    if b.startswith("sample_"):
        if "error" in b:
            # sample_errors.ayar has a deliberate LayoutError.
            return "sample", "parse_error"
        if b in ("sample_5.ayar", "sample_minimal.ayar"):
            # These contain unsplit-dataset errors by design.
            return "sample", "type_errors"
        return "sample", "clean"

    return "unknown", "skip"

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-FILE TEST EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(path: str):
    """
    Run one .ayar file through the full pipeline and evaluate against expectation.

    Returns:
        passed  (bool)
        detail  (str)   — human-readable description of what happened
        stdout  (str)   — captured stdout (may be empty)
    """
    src        = open(path, encoding="utf-8").read()
    filename   = os.path.basename(path)
    _, expect  = _classify(filename)

    # ── LEX ──────────────────────────────────────────────────────────────────
    lex_err = None
    try:
        _lex(src)
    except LexicalError as e:
        lex_err = e

    if lex_err is not None:
        if expect == "parse_error":
            return True, f"LexicalError (expected): {lex_err}", ""
        return False, f"Unexpected LexicalError: {lex_err}", ""

    # ── PARSE ─────────────────────────────────────────────────────────────────
    parse_err = None
    try:
        _parse(src)
    except (ParseError, LayoutError) as e:
        parse_err = e

    if parse_err is not None:
        if expect == "parse_error":
            return True, f"ParseError/LayoutError (expected): {str(parse_err)[:120]}", ""
        return False, f"Unexpected ParseError: {str(parse_err)[:120]}", ""

    if expect == "parse_error":
        return False, "Expected a parse / lex error but the file parsed cleanly.", ""

    # ── TYPE CHECK ────────────────────────────────────────────────────────────
    tc_errs = []
    try:
        tc_errs = _typecheck(src)
    except Exception as e:
        return False, f"Exception during type checking: {type(e).__name__}: {e}", ""

    if expect == "type_errors":
        if tc_errs:
            rules = [e.rule for e in tc_errs]
            return True, f"{len(tc_errs)} type error(s) as expected: {rules}", ""
        return False, "Expected ≥1 AyarTypeError but type checker returned 0 errors.", ""

    if tc_errs and expect == "clean":
        rules = [e.rule for e in tc_errs]
        return False, f"Unexpected type error(s): {rules}", ""

    # ── INTERPRET ─────────────────────────────────────────────────────────────
    stdout, run_exc = _run(src)

    if run_exc is not None:
        return False, f"Runtime exception: {type(run_exc).__name__}: {run_exc}", stdout

    # Clean run — any amount of stdout (including empty) is fine.
    lines   = len([l for l in stdout.splitlines() if l.strip()])
    has_out = lines > 0
    note    = f"{lines} output line(s)" if has_out else "no output (no report/evaluate stmts)"
    return True, f"Clean execution — {note}.", stdout

# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

_CAT_ORDER = ["lexer", "parser", "typechecker", "interpreter", "sample", "unknown"]

def _discover():
    """Return all .ayar files in HERE, grouped by category."""
    all_files = sorted(glob.glob(os.path.join(HERE, "*.ayar")))
    grouped   = {c: [] for c in _CAT_ORDER}
    for path in all_files:
        cat, exp = _classify(os.path.basename(path))
        grouped[cat].append((path, exp))
    return grouped

def _cat_label(cat: str) -> str:
    return {
        "lexer":       "Lexer",
        "parser":      "Parser",
        "typechecker": "Type Checker",
        "interpreter": "Interpreter",
        "sample":      "Sample Files",
        "unknown":     "Unknown / Skipped",
    }.get(cat, cat.title())

def _exp_tag(exp: str) -> str:
    return {
        "clean":       dim("expect: clean run"),
        "parse_error": dim("expect: parse error"),
        "type_errors": dim("expect: type errors"),
        "skip":        dim("expect: skip"),
    }.get(exp, dim(exp))

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results):
    """
    results items: (cat, filename, passed, detail, stdout, ms, skipped)
    """
    active  = [r for r in results if not r[6]]
    passed  = sum(1 for r in active if r[2])
    failed  = sum(1 for r in active if not r[2])
    skipped = sum(1 for r in results if r[6])
    total   = len(active)

    print()
    print(bold("━" * 62))
    parts = [green(f"{passed} passed"),
             red(f"{failed} failed") if failed else dim("0 failed")]
    if skipped:
        parts.append(yellow(f"{skipped} skipped"))
    parts.append(f"{total} total")
    print(bold("  RESULTS   " + "  /  ".join(parts)))

    # Per-category breakdown
    cats_seen = []
    for r in results:
        if r[0] not in cats_seen:
            cats_seen.append(r[0])

    print()
    print(bold("  BY CATEGORY"))
    for cat in cats_seen:
        cat_res = [r for r in results if r[0] == cat and not r[6]]
        if not cat_res:
            continue
        p = sum(1 for r in cat_res if r[2])
        f = sum(1 for r in cat_res if not r[2])
        label = _cat_label(cat)
        dots  = green("●") * p + (red("●") * f if f else "")
        note  = red(f"  ← {f} FAILED") if f else ""
        print(f"    {label:<18} {dots}{note}")

    # Failure details
    failures = [r for r in results if not r[6] and not r[2]]
    if failures:
        print()
        print(bold("  FAILURES"))
        for cat, filename, _, detail, stdout, ms, __ in failures:
            print(f"    {red('✗')} {filename}")
            for line in detail.splitlines()[:6]:
                print(f"        {dim(line)}")
    else:
        print()
        print(f"  {green('✓')} All tests passed.")

    print(bold("━" * 62))
    print()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print(bold("╔══════════════════════════════════════════════════════════╗"))
    print(bold("║        Ayar DSL — Auto-Discovery Test Runner            ║"))
    print(bold("╚══════════════════════════════════════════════════════════╝"))
    print(dim(f"  Directory : {HERE}"))
    print(dim(f"  Python    : {sys.version.split()[0]}"))

    cat_filter = ARGS.category.lower().strip()
    valid_cats = ("", "lexer", "parser", "typechecker", "interpreter", "sample")
    if cat_filter and cat_filter not in valid_cats:
        print(red(f"\n  ERROR: Unknown --category '{cat_filter}'."))
        print(f"  Valid values: {', '.join(v for v in valid_cats if v)}")
        sys.exit(1)

    if cat_filter:
        print(dim(f"  Filter    : --category {cat_filter}"))

    grouped = _discover()

    total_found = sum(len(v) for v in grouped.values())
    if total_found == 0:
        print(yellow("\n  WARNING: No .ayar files found. "
                     "Make sure your test files are in:\n  " + HERE))
        sys.exit(0)

    print(dim(f"  Found     : {total_found} .ayar file(s)\n"))

    results = []   # (cat, filename, passed, detail, stdout, ms, skipped)

    for cat in _CAT_ORDER:
        entries = grouped.get(cat, [])
        if not entries:
            continue

        # Apply --category filter — mark non-matching entries as skipped
        if cat_filter and cat != cat_filter:
            for path, exp in entries:
                results.append((cat, os.path.basename(path),
                                 None, "Skipped (--category filter)", "", 0, True))
            continue

        label = _cat_label(cat)
        bar   = "─" * max(1, 52 - len(label))
        print(cyan(bold(f"  ── {label} {bar}")))

        for path, exp in entries:
            filename = os.path.basename(path)

            if exp == "skip":
                print(f"  {filename:<45}  {yellow('SKIP')}  "
                      f"{dim('unrecognised prefix')}")
                results.append((cat, filename, None,
                                 "Unrecognised prefix", "", 0, True))
                continue

            t0 = time.perf_counter()
            try:
                passed, detail, stdout = _run_one(path)
            except Exception as e:
                passed = False
                detail = f"HARNESS EXCEPTION — {type(e).__name__}: {e}"
                stdout = ""
            ms = (time.perf_counter() - t0) * 1000

            badge   = green("PASS") if passed else red("FAIL")
            timing  = dim(f"  [{ms:.0f}ms]")
            etag    = _exp_tag(exp)

            print(f"  {filename:<45}  {badge}  {etag}{timing}")

            # Print detail when the test failed or --verbose is on
            if not passed or ARGS.verbose:
                for line in detail.splitlines():
                    print(f"      {dim(line)}")

            # Print stdout excerpt when --verbose is on
            if ARGS.verbose and stdout.strip():
                print(f"      {dim('─── stdout (' + str(stdout.count(chr(10))) + ' lines) ───────────────')}")
                for line in stdout.splitlines()[:25]:
                    print(f"      {dim(line)}")
                if stdout.count("\n") > 25:
                    print(f"      {dim('  … (truncated)')}")

            results.append((cat, filename, passed, detail, stdout, ms, False))

            if ARGS.stop_first and not passed:
                _print_summary(results)
                sys.exit(1)

        print()   # blank line between categories

    _print_summary(results)


if __name__ == "__main__":
    main()
