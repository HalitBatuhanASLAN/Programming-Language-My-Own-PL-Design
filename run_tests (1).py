#!/usr/bin/env python3
"""
run_tests.py
============
Test runner for the ayar DSL lexer + parser.

Modes
─────
  python3 run_tests.py                   – full built-in suite (recommended first run)
  python3 run_tests.py --file foo.ayar   – parse a specific .ayar source file
  python3 run_tests.py --tokens foo.ayar – show the token stream only (lex step)
  python3 run_tests.py --expr "3+4*2"   – parse a single expression string

Changes from D1 Design Specification Document
─────────────────────────────────────────────
  1. analyze_field_list: fields are NOT comma-separated; each field ends with
     its own ";" and the list is simply { <analyze_field> }.
  2. eval_target simple form now also accepts <identifier> "." <identifier>
     (dotted, without brackets), matching the D1 sample: comparison.best
  3. Layout rule: every "{" and "}" must be on its own line.
"""

import sys
import os
import argparse
import textwrap

sys.path.insert(0, os.path.dirname(__file__))

from ayar_lexer  import Lexer, LexicalError
from ayar_parser import Parser, ParseError, LayoutError, validate_layout

# ── ANSI colour codes ────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

GREEN  = lambda t: _c("32;1", t)
RED    = lambda t: _c("31;1", t)
YELLOW = lambda t: _c("33;1", t)
CYAN   = lambda t: _c("36;1", t)
BOLD   = lambda t: _c("1",    t)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def lex_and_parse(source: str):
    tokens = Lexer(source).tokenize()
    ast    = Parser(tokens).parse()
    return tokens, ast


def run_pass_test(label: str, source: str) -> bool:
    print(BOLD(f"\n{'─'*62}"))
    print(BOLD(f"[PASS] {label}"))
    print(f"{'─'*62}")
    try:
        _, ast = lex_and_parse(source)
        for line in Parser.dump_ast(ast).splitlines():
            print("  " + line)
        print(GREEN("  ✓ PASSED"))
        return True
    except (ParseError, LayoutError, LexicalError) as e:
        print(RED("  ✗ FAILED  (unexpected error)"))
        print(RED(str(e)))
        return False


def run_fail_test(label: str, source: str, fragment: str = "") -> bool:
    print(BOLD(f"\n{'─'*62}"))
    print(BOLD(f"[FAIL] {label}"))
    print(f"{'─'*62}")
    try:
        lex_and_parse(source)
        print(RED("  ✗ FAILED  (no error raised – parser should have rejected this)"))
        return False
    except (ParseError, LayoutError, LexicalError) as e:
        msg = str(e)
        if fragment and fragment not in msg:
            print(YELLOW(f"  ⚠ PARTIAL – error raised but expected fragment not found"))
            print(YELLOW(f"    wanted : {fragment!r}"))
            print(YELLOW(f"    got    : {msg[:300]}"))
            return False
        for line in msg.splitlines():
            print("  " + line)
        print(GREEN("  ✓ PASSED"))
        return True


# ═════════════════════════════════════════════════════════════════════════════
# PASS TEST CASES
# All sources use D1 layout: every '{' and '}' on its own line.
# All sources use D1 analyze style: no comma between analyze fields.
# ═════════════════════════════════════════════════════════════════════════════

PASS_TESTS = [

    # P1 ── Exact D1 sample program from the design specification document
    ("D1 sample program (exact from design spec)",
     textwrap.dedent("""\
        dataset iris = load("iris.csv");

        split iris into train(0.70), validation(0.15), test(0.15);

        model KNN knn_baseline
        {
            k = 3,
            distance = "euclidean"
        }

        model DecisionTree dt_deep
        {
            max_depth = 15,
            criterion = "gini"
        }

        experiment iris_comparison
        {
            train knn_baseline on iris.train;
            train dt_deep on iris.train;
            evaluate knn_baseline on iris.validation;
            evaluate dt_deep on iris.validation;
            collect metrics [accuracy, precision, recall, f1];
            analyze overfitting
            {
                threshold = 0.10;
                underfit_max_acc = 0.70;
            }
            compare by f1 higher_is_better;
            exclude if overfit or underfit;
            select best;
        }

        evaluate comparison.best on iris.test;

        report final
        {
            metrics = [accuracy, precision, recall, f1],
            show = overfitting_analysis
        }
     """)),

    # P2 ── Analyze with only threshold field
    ("Analyze with single field (threshold only)",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 3
        }
        experiment e
        {
            train k on d.train;
            collect metrics [accuracy];
            analyze overfitting
            {
                threshold = 0.05;
            }
            compare by accuracy higher_is_better;
            select best;
        }
     """)),

    # P3 ── Analyze with only underfit_max_acc field
    ("Analyze with single field (underfit_max_acc only)",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model SVM s
        {
            C = 1
        }
        experiment e
        {
            train s on d.train;
            collect metrics [f1];
            analyze overfitting
            {
                underfit_max_acc = 0.60;
            }
            compare by f1 higher_is_better;
            select best;
        }
     """)),

    # P4 ── Analyze with both fields (D1 style, no comma)
    ("Analyze with both fields, no comma between them",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 5
        }
        experiment e
        {
            train k on d.train;
            collect metrics [accuracy];
            analyze overfitting
            {
                threshold = 0.10;
                underfit_max_acc = 0.70;
            }
            compare by accuracy higher_is_better;
            select best;
        }
     """)),

    # P5 ── eval_target: dotted form WITHOUT brackets (D1 sample style)
    ("eval_target: dotted identifier without brackets (comparison.best)",
     textwrap.dedent("""\
        dataset d = load("test.csv");
        evaluate comparison.best on d.test;
     """)),

    # P6 ── eval_target: dotted form with any identifier as attr
    ("eval_target: dotted identifier (myExp.winner)",
     textwrap.dedent("""\
        dataset d = load("test.csv");
        evaluate myExp.winner on d.test;
     """)),

    # P7 ── eval_target: bracket form still works
    ("eval_target: bracket form [exp.best] still valid",
     textwrap.dedent("""\
        dataset d = load("test.csv");
        evaluate [main_exp.best] on d.test;
     """)),

    # P8 ── eval_target: plain identifier (no dot, no brackets) still works
    ("eval_target: plain identifier (no dot)",
     textwrap.dedent("""\
        dataset d = load("test.csv");
        model KNN k
        {
            k = 1
        }
        evaluate k on d.test;
     """)),

    # P9 ── All five model types
    ("All five model types",
     textwrap.dedent("""\
        model KNN knn
        {
            k = 5
        }
        model DecisionTree dt
        {
            max_depth = 10
        }
        model SVM svm
        {
            C = 1
        }
        model NaiveBayes nb
        {
            var_smoothing = 1
        }
        model LogisticRegression lr
        {
            max_iter = 100
        }
     """)),

    # P10 ── All four field value types
    ("Field values: int, float, string, bool",
     textwrap.dedent("""\
        model SVM s
        {
            C = 2,
            tol = 0.001,
            kernel = "poly",
            shrinking = true,
            verbose = false
        }
     """)),

    # P11 ── All four metric literals
    ("All four metric literals",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 3
        }
        experiment e
        {
            train k on d.train;
            collect metrics [accuracy, precision, recall, f1];
            compare by recall lower_is_better;
            select best;
        }
     """)),

    # P12 ── Report with string show value
    ("Report with string show value",
     textwrap.dedent("""\
        report summary
        {
            metrics = [recall],
            show = "confusion_matrix"
        }
     """)),

    # P13 ── exclude with single term
    ("Exclude with single term",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 3
        }
        experiment e
        {
            train k on d.train;
            collect metrics [accuracy];
            compare by accuracy higher_is_better;
            exclude if overfit;
            select best;
        }
     """)),

    # P14 ── compare lower_is_better
    ("Compare direction: lower_is_better",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model SVM s
        {
            C = 1
        }
        experiment e
        {
            train s on d.train;
            collect metrics [recall];
            compare by recall lower_is_better;
            select best;
        }
     """)),

    # P15 ── Dataset + split only (no model or experiment)
    ("Dataset declaration and split only",
     textwrap.dedent("""\
        dataset iris = load("data/iris.csv");
        split iris into train(0.70), validation(0.15), test(0.15);
     """)),
]


# ═════════════════════════════════════════════════════════════════════════════
# FAIL TEST CASES
# Each tuple: (label, source, fragment_that_must_appear_in_the_error_message)
# ═════════════════════════════════════════════════════════════════════════════

FAIL_TESTS = [

    # ── Grammar errors ─────────────────────────────────────────────────────────

    # F1 – missing semicolon after dataset
    ("Missing ';' after dataset declaration",
     'dataset iris = load("iris.csv")\nmodel KNN k\n{\n    k = 3\n}',
     "SEMICOLON"),

    # F2 – invalid model type
    ("Invalid model type 'RandomForest'",
     "model RandomForest rf\n{\n    n_estimators = 100\n}",
     "model type"),

    # F3 – invalid split role
    ("Invalid split role 'verify'",
     "split ds into train(0.6), verify(0.2), test(0.2);",
     "split role"),

    # F4 – missing OP_ASSIGN in model field
    ("Missing '=' in model field",
     "model KNN k\n{\n    neighbours 5\n}",
     "OP_ASSIGN"),

    # F5 – unknown top-level keyword
    ("Unknown top-level keyword 'pipeline'",
     "pipeline myPipeline\n{\n}",
     "top-level declaration"),

    # F6 – missing 'into' in split
    ("Missing 'into' in split statement",
     "split ds train(0.7), validation(0.15), test(0.15);",
     "into"),

    # F7 – invalid metric name
    ("Invalid metric 'loss' (not a METRIC_LITERAL)",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 3
        }
        experiment e
        {
            train k on d.train;
            collect metrics [accuracy, loss];
            compare by accuracy higher_is_better;
            select best;
        }
     """),
     "METRIC_LITERAL"),

    # F8 – missing DOT in dataset_ref
    ("Missing '.' in dataset reference",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 3
        }
        experiment e
        {
            train k on d train;
        }
     """),
     "DOT"),

    # F9 – unclosed experiment brace
    ("Unclosed experiment block (missing '}')",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 3
        }
        experiment e
        {
            train k on d.train;
            select best;
     """),
     "'}' to close experiment block"),

    # F10 – invalid compare direction
    ("Invalid compare direction 'max_is_better'",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 3
        }
        experiment e
        {
            train k on d.train;
            collect metrics [accuracy];
            compare by accuracy max_is_better;
            select best;
        }
     """),
     "comparison direction"),

    # F11 – BOOL_LITERAL where metric expected
    ("Bool literal where metric expected in collect",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 3
        }
        experiment e
        {
            train k on d.train;
            collect metrics [true];
            compare by accuracy higher_is_better;
            select best;
        }
     """),
     "METRIC_LITERAL"),

    # F12 – missing identifier before '='
    ("Missing identifier before '=' in field",
     "model KNN k\n{\n    = 5\n}",
     "IDENTIFIER"),

    # F13 – D1-specific: analyze field with comma separator (old wrong style)
    #       The comma is now NOT a separator; it would be seen as starting a
    #       new model field or as an unexpected token inside the analyze block.
    ("Analyze fields with comma between them (old style, now illegal)",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 3
        }
        experiment e
        {
            train k on d.train;
            collect metrics [accuracy];
            analyze overfitting
            {
                threshold = 0.10;,
                underfit_max_acc = 0.60;
            }
            compare by accuracy higher_is_better;
            select best;
        }
     """),
     # After 'threshold = 0.10;' the next token is COMMA then 'underfit_max_acc'.
     # The while loop stops (COMMA is not a keyword field name), then
     # _consume("RBRACE") is called but finds COMMA -> ParseError.
     "RBRACE"),

    # ── Layout errors ──────────────────────────────────────────────────────────

    # L1 – LBRACE on same line as model header
    ("Layout: '{' on same line as 'model KNN k'",
     "model KNN k {\n    n = 5\n}",
     "LayoutError"),

    # L2 – LBRACE on same line as experiment header
    ("Layout: '{' on same line as 'experiment e'",
     "dataset d = load(\"x.csv\");\nexperiment e {\n    select best;\n}",
     "LayoutError"),

    # L3 – LBRACE on same line as report header
    ("Layout: '{' on same line as 'report r'",
     "report r {\n    metrics = [accuracy],\n    show = overfitting_analysis\n}",
     "LayoutError"),

    # L4 – LBRACE on same line as analyze overfitting
    ("Layout: '{' on same line as 'analyze overfitting'",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            n = 5
        }
        experiment e
        {
            train k on d.train;
            analyze overfitting {
                threshold = 0.1;
            }
            select best;
        }
     """),
     "LayoutError"),

    # L5 – RBRACE on same line as last field content
    ("Layout: '}' on same line as field content",
     "model KNN k\n{\n    n = 5 }",
     "LayoutError"),

    # L6 – single-line block
    ("Layout: single-line block '{ n = 5 }'",
     "model KNN k { n = 5 }",
     "LayoutError"),
]


# ═════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_builtin_suite():
    passed = failed = 0

    print(BOLD("\n" + "═"*62))
    print(BOLD("  ayar DSL – Parser Test Suite  (D1 spec aligned)"))
    print(BOLD("  Lexer : student's own ayar_lexer.py"))
    print(BOLD("  Rules : analyze fields comma-free | dotted eval_target | layout"))
    print(BOLD("═"*62))

    print(BOLD("\n── PASS tests (must succeed) ──────────────────────────────"))
    for label, source in PASS_TESTS:
        ok = run_pass_test(label, source)
        if ok: passed += 1
        else:  failed += 1

    print(BOLD("\n── FAIL tests (must raise an error) ───────────────────────"))
    for label, source, frag in FAIL_TESTS:
        ok = run_fail_test(label, source, frag)
        if ok: passed += 1
        else:  failed += 1

    total = passed + failed
    print(BOLD("\n" + "═"*62))
    summary = f"  Results: {passed}/{total} tests passed"
    print(GREEN(summary) if failed == 0 else RED(summary + f"  ({failed} FAILED)"))
    print(BOLD("═"*62 + "\n"))


def run_file(path: str):
    print(BOLD(f"\nParsing: {path}"))
    try:
        source = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(RED(f"File not found: {path}")); sys.exit(1)
    try:
        tokens = Lexer(source).tokenize()
        ast    = Parser(tokens).parse()
        print(CYAN("\nAST:"))
        for line in Parser.dump_ast(ast).splitlines():
            print("  " + line)
        print(GREEN("\n✓ Parse succeeded."))
    except LexicalError  as e: print(RED(f"\n✗ Lex error:{e}"));    sys.exit(1)
    except LayoutError   as e: print(RED(f"\n✗ Layout error:{e}")); sys.exit(1)
    except ParseError    as e: print(RED(f"\n✗ Parse error:{e}"));  sys.exit(1)


def run_tokens_only(path: str):
    try:
        source = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(RED(f"File not found: {path}")); sys.exit(1)
    tokens = Lexer(source).tokenize()
    print(BOLD(f"\nToken stream for: {path}"))
    w_type  = max(len(t.type)  for t in tokens) + 2
    w_value = max(len(t.value) for t in tokens) + 2
    hdr = f"  {'TYPE':<{w_type}} {'VALUE':<{w_value}} {'LINE':>5}  {'COL':>5}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for t in tokens:
        print(f"  {t.type:<{w_type}} {t.value!r:<{w_value}} "
              f"{t.line_number:>5}  {t.column_number:>5}")


def run_expr(expr_source: str):
    print(BOLD(f"\nParsing expression: {expr_source!r}"))
    try:
        tokens = Lexer(expr_source).tokenize()
        ast    = Parser(tokens).parse_expr()
        print(CYAN("  AST:"))
        for line in Parser.dump_ast(ast).splitlines():
            print("  " + line)
        print(GREEN("  ✓ Parsed successfully."))
    except (LexicalError, LayoutError, ParseError) as e:
        print(RED(f"  ✗ Error: {e}"))


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Test runner for the ayar DSL parser.")
    ap.add_argument("--file",   metavar="PATH", help="Parse a .ayar file and print its AST.")
    ap.add_argument("--tokens", metavar="PATH", help="Show the token stream for a .ayar file.")
    ap.add_argument("--expr",   metavar="EXPR", help='Parse a single expression, e.g. --expr "3+4*2"')
    args = ap.parse_args()

    if   args.tokens: run_tokens_only(args.tokens)
    elif args.file:   run_file(args.file)
    elif args.expr:   run_expr(args.expr)
    else:             run_builtin_suite()


if __name__ == "__main__":
    main()
