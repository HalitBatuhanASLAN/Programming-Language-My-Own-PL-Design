#!/usr/bin/env python3
"""
run_tests.py  –  Test runner for the ayar DSL lexer + parser
═════════════════════════════════════════════════════════════════════════════
CSE341 – Concepts of Programming Languages  |  Deliverable D3

USAGE
─────
  python3 run_tests.py                   full built-in test suite
  python3 run_tests.py --file  my.ayar   parse a .ayar file, print AST
  python3 run_tests.py --tokens my.ayar  show the raw token stream only
  python3 run_tests.py --expr "3+4*2"    parse a single expression

WHAT CHANGED FROM THE PREVIOUS PARSER (exam-ready explanation)
───────────────────────────────────────────────────────────────
  1. analyze overfitting fields  – NO comma between them.
     Each field is self-terminating with ";", so two fields look like:
         analyze overfitting
         {
             threshold = 0.10;
             underfit_max_acc = 0.70;
         }
     Using the old ";," style now causes a ParseError because the parser
     sees the "}" but finds a COMMA instead after the first field's ";".

  2. eval_target (top-level evaluate) – three forms now accepted:
       a) plain identifier        evaluate my_model on d.test;
       b) dotted (NEW)            evaluate exp_name.best on d.test;
       c) bracket                 evaluate [exp_name.best] on d.test;
     The dotted form is new. The bracket form still works.

TOKEN ADAPTER (adapt_tokens)
─────────────────────────────
  The lexer and parser agree on all token-type names, so REMAP is empty.
  The function is kept as architecture: if a future name mismatch occurs,
  add  "old_name": "new_name"  to REMAP and nothing else needs changing.

LAYOUT RULE
────────────
  Every "{" and "}" must be on its own source line. Violations raise
  LayoutError before grammar parsing even begins.
"""

import sys
import os
import argparse
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ayar_lexer  import Lexer, LexicalError
from ayar_parser import Parser, ParseError, LayoutError

# ── ANSI colour helpers ──────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text
GREEN  = lambda t: _c("32;1", t)
RED    = lambda t: _c("31;1", t)
YELLOW = lambda t: _c("33;1", t)
CYAN   = lambda t: _c("36;1", t)
BOLD   = lambda t: _c("1",    t)


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN ADAPTER
# ═════════════════════════════════════════════════════════════════════════════

REMAP: dict = {
    # Add entries here if the lexer renames a token type, e.g.:
    #   "STRING_LIT": "STRING_LITERAL"
}

def adapt_tokens(tokens: list) -> list:
    """
    Interceptor that fixes token-type name mismatches between lexer and parser.
    Walks the token list once (O(n)), rewrites any type found in REMAP.
    Currently a no-op (REMAP is empty) but kept as future-proof architecture.
    """
    for tok in tokens:
        if tok.type in REMAP:
            tok.type = REMAP[tok.type]
    return tokens


# ═════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def lex_and_parse(source: str):
    """Tokenise → adapt → parse.  Returns (tokens, ast)."""
    tokens = Lexer(source).tokenize()
    tokens = adapt_tokens(tokens)
    ast    = Parser(tokens).parse()
    return tokens, ast


# ═════════════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def run_valid_test(label: str, source: str) -> bool:
    """Parse source; PASS if no error, FAIL if any error is raised."""
    print(BOLD(f"\n{'─'*64}"))
    print(BOLD(f"[VALID] {label}"))
    print(f"{'─'*64}")
    try:
        _, ast = lex_and_parse(source)
        for line in Parser.dump_ast(ast).splitlines():
            print("  " + line)
        print(GREEN("  ✓ PASSED"))
        return True
    except (LexicalError, LayoutError, ParseError) as err:
        print(RED("  ✗ FAILED  (unexpected error raised)"))
        for line in str(err).splitlines():
            print(RED("  " + line))
        return False


def run_invalid_test(label: str, source: str, expected_fragment: str = "") -> bool:
    """
    Parse source; PASS if LexicalError / LayoutError / ParseError is raised.
    If expected_fragment is given, that string must appear in the error message.
    """
    print(BOLD(f"\n{'─'*64}"))
    print(BOLD(f"[INVALID] {label}"))
    print(f"{'─'*64}")
    try:
        lex_and_parse(source)
        print(RED("  ✗ FAILED  (no error raised — parser accepted invalid input)"))
        return False
    except (LexicalError, LayoutError, ParseError) as err:
        msg = str(err)
        if expected_fragment and expected_fragment not in msg:
            print(YELLOW("  ⚠ PARTIAL – error raised but expected fragment missing"))
            print(YELLOW(f"    wanted : {expected_fragment!r}"))
            print(YELLOW(f"    got    : {msg[:300]}"))
            return False
        for line in msg.splitlines():
            print("  " + line)
        print(GREEN("  ✓ PASSED"))
        return True


# ═════════════════════════════════════════════════════════════════════════════
# VALID TEST CASES
# ─────────────────────────────────────────────────────────────────────────────
# All analyze blocks use the NEW style: fields separated by nothing,
# each ending with ";" only.
# eval_target tests cover all three accepted forms.
# ═════════════════════════════════════════════════════════════════════════════

def _sample_file(name):
    """Load a .ayar file from the same directory as this script, or '' if missing."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    return open(path).read() if os.path.exists(path) else ""


VALID_TESTS = [

    # ── V1 : dataset declaration + three-way split ───────────────────────────
    ("V1 – dataset decl and three-way split",
     textwrap.dedent("""\
        dataset iris = load("data/iris.csv");
        split iris into
            train(0.70),
            validation(0.15),
            test(0.15);
     """)),

    # ── V2 : all five model types ─────────────────────────────────────────────
    ("V2 – all five model types declared",
     textwrap.dedent("""\
        model KNN knn
        {
            n_neighbors = 5
        }
        model DecisionTree dt
        {
            max_depth = 8
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

    # ── V3 : all four field value types ──────────────────────────────────────
    ("V3 – field values: int, float, string, bool",
     textwrap.dedent("""\
        model SVM s
        {
            C = 2,
            tol = 0.001,
            kernel = "poly",
            probability = true,
            shrinking = false
        }
     """)),

    # ── V4 : analyze with BOTH fields, no comma (NEW style) ──────────────────
    ("V4 – analyze overfitting: both fields, no comma between them",
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

    # ── V5 : analyze with only threshold field ────────────────────────────────
    ("V5 – analyze overfitting: threshold field only",
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

    # ── V6 : analyze with only underfit_max_acc field ─────────────────────────
    ("V6 – analyze overfitting: underfit_max_acc field only",
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

    # ── V7 : all seven experiment statement types ─────────────────────────────
    ("V7 – all seven experiment statement types",
     textwrap.dedent("""\
        dataset d = load("bank.csv");
        model KNN k
        {
            k = 7
        }
        experiment full_exp
        {
            train    k on d.train;
            evaluate k on d.validation;
            collect metrics [accuracy, precision, recall, f1];
            analyze overfitting
            {
                threshold = 0.10;
                underfit_max_acc = 0.55;
            }
            compare by f1 higher_is_better;
            exclude if overfit or underfit;
            select best;
        }
     """)),

    # ── V8 : eval_target – dotted form (NEW) ──────────────────────────────────
    ("V8 – eval_target: dotted form  evaluate exp.best on d.test",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        evaluate iris_exp.best on d.test;
     """)),

    # ── V9 : eval_target – bracket form (still valid) ─────────────────────────
    ("V9 – eval_target: bracket form  evaluate [exp.best] on d.test",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        evaluate [main_exp.best] on d.test;
     """)),

    # ── V10 : eval_target – plain identifier ─────────────────────────────────
    ("V10 – eval_target: plain identifier  evaluate model_name on d.test",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 1
        }
        evaluate k on d.test;
     """)),

    # ── V11 : compare direction lower_is_better ───────────────────────────────
    ("V11 – compare direction: lower_is_better",
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

    # ── V12 : exclude with single condition ──────────────────────────────────
    ("V12 – exclude if overfit (single condition)",
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

    # ── V13 : all four metric literals ───────────────────────────────────────
    ("V13 – collect all four metric literals",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model DecisionTree dt
        {
            max_depth = 5
        }
        experiment e
        {
            train dt on d.train;
            collect metrics [accuracy, precision, recall, f1];
            compare by accuracy higher_is_better;
            select best;
        }
     """)),

    # ── V14 : report – overfitting_analysis show value ────────────────────────
    ("V14 – report with show = overfitting_analysis",
     textwrap.dedent("""\
        report summary
        {
            metrics = [accuracy, f1],
            show    = overfitting_analysis
        }
     """)),

    # ── V15 : report – string literal show value ──────────────────────────────
    ("V15 – report with show = \"confusion_matrix\" (string literal)",
     textwrap.dedent("""\
        report summary
        {
            metrics = [recall, precision],
            show    = "confusion_matrix"
        }
     """)),

    # ── V16 : sample_1.ayar (KNN + DecisionTree, iris, dotted eval_target) ───
    ("V16 – sample_1.ayar: KNN + DecisionTree on iris, dotted eval_target",
     _sample_file("sample_1.ayar") or textwrap.dedent("""\
        dataset iris = load("data/iris.csv");
        split iris into train(0.70), validation(0.15), test(0.15);
        model KNN knn_clf
        {
            n_neighbors = 7,
            weights = "distance"
        }
        model DecisionTree dt_clf
        {
            max_depth = 8,
            criterion = "gini"
        }
        experiment iris_exp
        {
            train knn_clf on iris.train;
            train dt_clf  on iris.train;
            evaluate knn_clf on iris.validation;
            evaluate dt_clf  on iris.validation;
            collect metrics [accuracy, precision, recall, f1];
            analyze overfitting
            {
                threshold = 0.08;
                underfit_max_acc = 0.55;
            }
            compare by f1 higher_is_better;
            exclude if overfit or underfit;
            select best;
        }
        evaluate iris_exp.best on iris.test;
        report iris_report
        {
            metrics = [accuracy, precision, recall, f1],
            show    = overfitting_analysis
        }
     """)),

    # ── V17 : sample_2.ayar (SVM + NaiveBayes + LR, bracket eval_target) ────
    ("V17 – sample_2.ayar: SVM + NaiveBayes + LR on spam, bracket eval_target",
     _sample_file("sample_2.ayar") or textwrap.dedent("""\
        dataset spam = load("datasets/spam_emails.csv");
        split spam into train(0.65), validation(0.20), test(0.15);
        model SVM svm_rbf
        {
            C = 10,
            kernel = "rbf",
            probability = true
        }
        model NaiveBayes gnb
        {
            var_smoothing = 1
        }
        model LogisticRegression lr_spam
        {
            C = 1,
            max_iter = 500,
            solver = "liblinear"
        }
        experiment spam_exp
        {
            train svm_rbf on spam.train;
            train gnb     on spam.train;
            train lr_spam on spam.train;
            evaluate svm_rbf on spam.validation;
            evaluate gnb     on spam.validation;
            evaluate lr_spam on spam.validation;
            collect metrics [precision, recall, f1];
            analyze overfitting
            {
                threshold = 0.10;
            }
            compare by recall lower_is_better;
            exclude if underfit;
            select best;
        }
        evaluate [spam_exp.best] on spam.test;
        report spam_report
        {
            metrics = [precision, recall, f1],
            show    = "confusion_matrix"
        }
     """)),

    # ── V18 : sample_3.ayar (all five models, churn, plain eval_target) ──────
    ("V18 – sample_3.ayar: all five models on churn, plain eval_target",
     _sample_file("sample_3.ayar") or textwrap.dedent("""\
        dataset churn = load("data/customer_churn.csv");
        split churn into train(0.70), validation(0.10), test(0.20);
        model KNN knn_churn
        {
            n_neighbors = 5
        }
        model DecisionTree dt_churn
        {
            max_depth = 12,
            criterion = "entropy"
        }
        model SVM svm_churn
        {
            C = 5,
            kernel = "linear"
        }
        model NaiveBayes nb_churn
        {
            var_smoothing = 1
        }
        model LogisticRegression lr_churn
        {
            C = 1,
            max_iter = 200,
            solver = "lbfgs"
        }
        experiment churn_exp
        {
            train knn_churn on churn.train;
            train dt_churn  on churn.train;
            train svm_churn on churn.train;
            train nb_churn  on churn.train;
            train lr_churn  on churn.train;
            evaluate knn_churn on churn.validation;
            evaluate dt_churn  on churn.validation;
            evaluate svm_churn on churn.validation;
            evaluate nb_churn  on churn.validation;
            evaluate lr_churn  on churn.validation;
            collect metrics [accuracy, precision, recall, f1];
            analyze overfitting
            {
                threshold = 0.05;
                underfit_max_acc = 0.60;
            }
            compare by accuracy higher_is_better;
            exclude if overfit or underfit;
            select best;
        }
        evaluate knn_churn on churn.test;
        report churn_report
        {
            metrics = [accuracy, precision, recall, f1],
            show    = overfitting_analysis
        }
     """)),
]


# ═════════════════════════════════════════════════════════════════════════════
# INVALID TEST CASES
# ─────────────────────────────────────────────────────────────────────────────
# Each tuple: (label, source, fragment_that_must_appear_in_error_message)
# ═════════════════════════════════════════════════════════════════════════════

INVALID_TESTS = [

    # ── Grammar errors ────────────────────────────────────────────────────────

    # I1 – comma between analyze fields (OLD style, now illegal)
    ("I1 – comma between analyze fields is now a ParseError (old ;, style)",
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
     "RBRACE"),

    # I2 – missing ';' after dataset declaration
    ("I2 – missing ';' after dataset declaration (ParseError)",
     textwrap.dedent("""\
        dataset titanic = load("data/titanic.csv")

        model KNN k
        {
            k = 3
        }
     """),
     "SEMICOLON"),

    # I3 – unknown model type 'RandomForest'
    ("I3 – unknown model type 'RandomForest' (ParseError)",
     textwrap.dedent("""\
        dataset bank = load("data/bank.csv");
        model RandomForest rf_model
        {
            n_estimators = 100,
            max_depth = 10
        }
     """),
     "model type"),

    # I4 – invalid split role 'verify'
    ("I4 – invalid split role 'verify' (ParseError)",
     textwrap.dedent("""\
        dataset wine = load("data/wine.csv");
        split wine into
            train(0.60),
            verify(0.20),
            test(0.20);
     """),
     "split role"),

    # I5 – missing '=' in model field
    ("I5 – missing '=' in model field (ParseError)",
     textwrap.dedent("""\
        model KNN k
        {
            neighbours 5
        }
     """),
     "OP_ASSIGN"),

    # I6 – unknown top-level keyword 'pipeline'
    ("I6 – unknown top-level keyword 'pipeline' (ParseError)",
     textwrap.dedent("""\
        dataset credit = load("data/credit_card.csv");
        model LogisticRegression lr_credit
        {
            C = 1,
            max_iter = 300
        }
        pipeline credit_pipeline
        {
            train lr_credit on credit.train;
            select best;
        }
     """),
     "top-level declaration"),

    # I7 – invalid metric name 'loss'
    ("I7 – invalid metric name 'loss' in collect (ParseError)",
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

    # I8 – missing '.' in dataset reference
    ("I8 – missing '.' in dataset reference (ParseError)",
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

    # I9 – unclosed experiment block
    ("I9 – unclosed experiment block, missing '}' (ParseError)",
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

    # I10 – invalid compare direction
    ("I10 – invalid compare direction 'max_is_better' (ParseError)",
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

    # I11 – analyze block with zero fields (empty braces)
    ("I11 – analyze overfitting with empty body (ParseError)",
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
            }
            compare by accuracy higher_is_better;
            select best;
        }
     """),
     "at least one analyze field"),

    # ── Layout errors ─────────────────────────────────────────────────────────

    # I12 – '{' on same line as model header
    ("I12 – layout: '{' on same line as model header (LayoutError)",
     "model KNN k {\n    n = 5\n}",
     "LayoutError"),

    # I13 – '{' on same line as experiment header
    ("I13 – layout: '{' on same line as experiment header (LayoutError)",
     'dataset d = load("x.csv");\nexperiment e {\n    select best;\n}',
     "LayoutError"),

    # I14 – '{' on same line as report header
    ("I14 – layout: '{' on same line as report header (LayoutError)",
     "report r {\n    metrics = [accuracy],\n    show = overfitting_analysis\n}",
     "LayoutError"),

    # I15 – '{' on same line as analyze overfitting
    ("I15 – layout: '{' on same line as analyze overfitting (LayoutError)",
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

    # I16 – '}' on same line as field content
    ("I16 – layout: '}' on same line as field content (LayoutError)",
     "model KNN k\n{\n    n = 5 }",
     "LayoutError"),
]


# ═════════════════════════════════════════════════════════════════════════════
# TEST RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

def run_valid_tests():
    passed = failed = 0
    print(BOLD("\n── VALID tests (must parse without errors) " + "─" * 20))
    for label, source in VALID_TESTS:
        ok = run_valid_test(label, source)
        passed += ok; failed += not ok
    return passed, failed


def run_invalid_tests():
    passed = failed = 0
    print(BOLD("\n── INVALID tests (must raise an error) " + "─" * 23))
    for label, source, fragment in INVALID_TESTS:
        ok = run_invalid_test(label, source, fragment)
        passed += ok; failed += not ok
    return passed, failed


def run_builtin_suite():
    print(BOLD("\n" + "═" * 64))
    print(BOLD("  ayar DSL – D3 Test Suite  (updated parser / D1 spec)"))
    print(BOLD("  Lexer  : ayar_lexer.py  (unmodified)"))
    print(BOLD("  Parser : ayar_parser.py  (updated)"))
    print(BOLD("  Change : analyze fields use ';' only, no comma between"))
    print(BOLD("  Change : eval_target accepts dotted form (exp.best)"))
    print(BOLD("═" * 64))

    vp, vf = run_valid_tests()
    ip, if_ = run_invalid_tests()

    total_p = vp + ip
    total_f = vf + if_
    total   = total_p + total_f

    print(BOLD("\n" + "═" * 64))
    print(BOLD(f"  Valid   tests : {vp}/{vp+vf} passed"))
    print(BOLD(f"  Invalid tests : {ip}/{ip+if_} passed"))
    summary = f"  Total         : {total_p}/{total} passed"
    print(GREEN(summary) if total_f == 0 else RED(summary + f"  ({total_f} FAILED)"))
    print(BOLD("═" * 64 + "\n"))


# ═════════════════════════════════════════════════════════════════════════════
# CLI UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def run_file(path: str):
    print(BOLD(f"\nParsing file: {path}"))
    try:
        source = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(RED(f"File not found: {path}")); sys.exit(1)
    try:
        tokens = Lexer(source).tokenize()
        tokens = adapt_tokens(tokens)
        ast    = Parser(tokens).parse()
        print(CYAN("\nAST:"))
        for line in Parser.dump_ast(ast).splitlines():
            print("  " + line)
        print(GREEN("\n✓ Parse succeeded."))
    except LexicalError as e:
        print(RED(f"\n✗ LexicalError: {e}")); sys.exit(1)
    except LayoutError as e:
        print(RED(f"\n✗ LayoutError: {e}")); sys.exit(1)
    except ParseError as e:
        print(RED(f"\n✗ ParseError: {e}")); sys.exit(1)


def run_tokens_only(path: str):
    try:
        source = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(RED(f"File not found: {path}")); sys.exit(1)
    tokens = Lexer(source).tokenize()
    tokens = adapt_tokens(tokens)
    print(BOLD(f"\nToken stream for: {path}"))
    w_type  = max(len(t.type)  for t in tokens) + 2
    w_value = max(len(t.value) for t in tokens) + 2
    hdr = f"  {'TYPE':<{w_type}} {'VALUE':<{w_value}} {'LINE':>5}  {'COL':>5}"
    print(hdr); print("  " + "─" * (len(hdr) - 2))
    for tok in tokens:
        print(f"  {tok.type:<{w_type}} {tok.value!r:<{w_value}} "
              f"{tok.line_number:>5}  {tok.column_number:>5}")


def run_expr(expr_source: str):
    print(BOLD(f"\nParsing expression: {expr_source!r}"))
    try:
        tokens = adapt_tokens(Lexer(expr_source).tokenize())
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
    ap = argparse.ArgumentParser(description="Test runner for the ayar DSL (CSE341 D3).")
    ap.add_argument("--file",   metavar="PATH", help="Parse a .ayar file and print its AST.")
    ap.add_argument("--tokens", metavar="PATH", help="Show the token stream for a .ayar file.")
    ap.add_argument("--expr",   metavar="EXPR", help='Parse a single expression.')
    args = ap.parse_args()
    if   args.tokens: run_tokens_only(args.tokens)
    elif args.file:   run_file(args.file)
    elif args.expr:   run_expr(args.expr)
    else:             run_builtin_suite()

if __name__ == "__main__":
    main()