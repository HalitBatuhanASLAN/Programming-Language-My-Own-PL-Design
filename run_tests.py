#!/usr/bin/env python3
"""
run_tests.py  –  Test runner for the ayar DSL lexer + parser
═════════════════════════════════════════════════════════════════════════════
CSE341 – Concepts of Programming Languages  |  Deliverable D3

HOW TO USE
──────────
  python3 run_tests.py                        full built-in test suite
  python3 run_tests.py --file  my.ayar        parse a .ayar file, print AST
  python3 run_tests.py --tokens my.ayar       show the token stream only
  python3 run_tests.py --expr "3+4*2"         parse & print a single expression

DESIGN OVERVIEW (for exam purposes)
────────────────────────────────────
  1. We import the student's own Lexer (ayar_lexer.py) and Parser
     (ayar_parser.py) without modifying either file.

  2. adapt_tokens(tokens)  –  an adapter / interceptor function.
     If in a future revision the lexer changes a token name (e.g. renames
     "OP_ASSIGN" to "ASSIGN" or "STRING_LITERAL" to "STRING_LIT"), this
     single function is the ONLY place that needs editing.  It loops through
     the token list and rewrites type names before handing them to the Parser.
     Currently the lexer and parser already agree on all names, so the
     REMAP dict below is empty — but the function is included so the
     architecture is clear and easy to extend.

  3. run_valid_tests()   iterates VALID_TESTS, calls lex → adapt → parse,
     and prints the AST with Parser.print_tree().  Any unexpected error
     counts as a test failure.

  4. run_invalid_tests() iterates INVALID_TESTS, calls lex → adapt → parse,
     and expects a LexicalError, LayoutError, or ParseError to be raised.
     If no exception is raised the test fails (the parser should have
     rejected the bad input).

  5. A simple counter tracks passed / failed.  No external framework is
     used — just plain Python try/except blocks and print() calls.

TOKEN NAME MISMATCH NOTES
──────────────────────────
  As of this writing, the lexer and parser agree on every token type name,
  so the REMAP table is empty.  The adapter is here to show WHERE fixes
  would go.  For example, if the lexer were changed to emit "ASSIGN" instead
  of "OP_ASSIGN", you would add:
      "ASSIGN": "OP_ASSIGN"
  to the REMAP dict below — nothing else in the codebase would need to change.

LAYOUT RULE (from the grammar spec)
─────────────────────────────────────
  Every '{' and '}' must be on its own source line.
  LEGAL:    model KNN k       ILLEGAL:  model KNN k {
            {                               n = 5
                n = 5                   }
            }
"""

import sys
import os
import argparse
import textwrap

# ── make sure ayar_lexer.py / ayar_parser.py are importable ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ayar_lexer  import Lexer, LexicalError
from ayar_parser import Parser, ParseError, LayoutError

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers  (gracefully disabled when output is not a terminal)
# ─────────────────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

GREEN  = lambda t: _c("32;1", t)
RED    = lambda t: _c("31;1", t)
YELLOW = lambda t: _c("33;1", t)
CYAN   = lambda t: _c("36;1", t)
BOLD   = lambda t: _c("1",    t)


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN ADAPTER / INTERCEPTOR
# ─────────────────────────────────────────────────────────────────────────────
# If the lexer and parser ever disagree on a token-type name, add the mapping
# here.  The function mutates the token list IN PLACE (no copy needed because
# Token is a mutable dataclass).
#
# REMAP maps  "lexer produces"  →  "parser expects".
# ═════════════════════════════════════════════════════════════════════════════

REMAP: dict = {
    # Example (not currently needed, shown for clarity):
    #   "ASSIGN"       : "OP_ASSIGN",     # if lexer renamed the assign token
    #   "STRING_LIT"   : "STRING_LITERAL", # if lexer shortened the string type
    #   "INT_LIT"      : "INT_LITERAL",    # if lexer shortened the int type
}


def adapt_tokens(tokens: list) -> list:
    """
    Adapter / interceptor that fixes token-type name mismatches between the
    lexer and the parser.

    Walks the entire token list exactly once (O(n)) and replaces any type
    name that appears in the REMAP dict.  Returns the same list object
    (the mutation is in-place for efficiency, but callers may also chain:
         ast = Parser(adapt_tokens(tokens)).parse()
    ).

    The adapter is intentionally separate from both the Lexer and the Parser
    so that neither file needs to be touched when a naming inconsistency is
    discovered.

    Parameters
    ----------
    tokens : list[Token]   – output of Lexer.tokenize()

    Returns
    -------
    list[Token]            – same list, with type names corrected
    """
    for tok in tokens:
        if tok.type in REMAP:
            tok.type = REMAP[tok.type]
    return tokens


# ═════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE  (lex → adapt → parse)
# ═════════════════════════════════════════════════════════════════════════════

def lex_and_parse(source: str):
    """Tokenise, adapt, and parse a source string.  Returns (tokens, ast)."""
    tokens = Lexer(source).tokenize()
    tokens = adapt_tokens(tokens)         # ← adapter runs here
    ast    = Parser(tokens).parse()
    return tokens, ast


# ═════════════════════════════════════════════════════════════════════════════
# GENERIC TEST HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def run_valid_test(label: str, source: str) -> bool:
    """
    Test that a source string parses without errors.

    On success  – prints the AST using Parser.print_tree().
    On failure  – prints the unexpected error message.

    Returns True if the test passed, False otherwise.
    """
    print(BOLD(f"\n{'─' * 64}"))
    print(BOLD(f"[VALID] {label}"))
    print(f"{'─' * 64}")
    try:
        _, ast = lex_and_parse(source)
        # print_tree uses Parser.dump_ast() which calls node.dump(indent=0)
        for line in Parser.dump_ast(ast).splitlines():
            print("  " + line)
        print(GREEN("  ✓ PASSED"))
        return True
    except (LexicalError, LayoutError, ParseError) as err:
        print(RED("  ✗ FAILED  (unexpected error was raised)"))
        for line in str(err).splitlines():
            print(RED("  " + line))
        return False


def run_invalid_test(label: str, source: str, expected_fragment: str = "") -> bool:
    """
    Test that a source string is REJECTED by the lexer/parser.

    The test passes if LexicalError, LayoutError, or ParseError is raised.
    If expected_fragment is non-empty, the error message must contain it.

    On success  – prints the caught error message (so you can see line/col).
    On failure  – prints a message explaining why the test failed.

    Returns True if the test passed, False otherwise.
    """
    print(BOLD(f"\n{'─' * 64}"))
    print(BOLD(f"[INVALID] {label}"))
    print(f"{'─' * 64}")
    try:
        lex_and_parse(source)
        # If we get here the parser accepted input it should have rejected.
        print(RED("  ✗ FAILED  (no error raised — parser should have rejected this)"))
        return False
    except (LexicalError, LayoutError, ParseError) as err:
        msg = str(err)
        if expected_fragment and expected_fragment not in msg:
            print(YELLOW("  ⚠ PARTIAL – correct error type raised, but expected "
                         "fragment NOT found in message"))
            print(YELLOW(f"    wanted : {expected_fragment!r}"))
            print(YELLOW(f"    got    : {msg[:300]}"))
            return False
        for line in msg.splitlines():
            print("  " + line)
        print(GREEN("  ✓ PASSED"))
        return True


# ═════════════════════════════════════════════════════════════════════════════
# VALID TEST CASES  (sample programs that MUST parse successfully)
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is a (label, source) tuple.
# The three sample .ayar files are also tested via run_file() below; these
# inline tests cover every individual grammar feature in isolation.
# ═════════════════════════════════════════════════════════════════════════════

VALID_TESTS = [

    # ── V1 : dataset declaration + split ─────────────────────────────────────
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

    # ── V3 : field value types – int, float, string, bool ────────────────────
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

    # ── V4 : all seven experiment statement types ─────────────────────────────
    ("V4 – all seven experiment statement types",
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
                threshold = 0.10;,
                underfit_max_acc = 0.55;
            }
            compare by f1 higher_is_better;
            exclude if overfit or underfit;
            select best;
        }
     """)),

    # ── V5 : top-level evaluate – bracket target ──────────────────────────────
    ("V5 – top-level evaluate with bracket target [exp.best]",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        evaluate [main_exp.best] on d.test;
     """)),

    # ── V6 : top-level evaluate – simple identifier target ────────────────────
    ("V6 – top-level evaluate with simple identifier target",
     textwrap.dedent("""\
        dataset d = load("x.csv");
        model KNN k
        {
            k = 1
        }
        evaluate k on d.test;
     """)),

    # ── V7 : report – overfitting_analysis show value ─────────────────────────
    ("V7 – report with show = overfitting_analysis",
     textwrap.dedent("""\
        report summary
        {
            metrics = [accuracy, f1],
            show    = overfitting_analysis
        }
     """)),

    # ── V8 : report – string literal show value ───────────────────────────────
    ("V8 – report with show = \"confusion_matrix\" (string literal)",
     textwrap.dedent("""\
        report summary
        {
            metrics = [recall, precision],
            show    = "confusion_matrix"
        }
     """)),

    # ── V9 : compare direction lower_is_better ────────────────────────────────
    ("V9 – compare by metric lower_is_better",
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

    # ── V10 : exclude with single condition ──────────────────────────────────
    ("V10 – exclude with single condition (exclude if overfit)",
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

    # ── V11 : analyze with single field ──────────────────────────────────────
    ("V11 – analyze overfitting with only the threshold field",
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

    # ── V12 : all four metric literals in collect ─────────────────────────────
    ("V12 – collect metrics [accuracy, precision, recall, f1]",
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

    # ── V13 : sample_1.ayar inline (KNN + DecisionTree / iris) ───────────────
    ("V13 – sample_1 inline: KNN + DecisionTree, iris dataset",
     open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "sample_1.ayar")).read()
     if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "sample_1.ayar")) else
     textwrap.dedent("""\
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
                threshold = 0.08;,
                underfit_max_acc = 0.55;
            }
            compare by f1 higher_is_better;
            exclude if overfit or underfit;
            select best;
        }
        evaluate [iris_exp.best] on iris.test;
        report iris_report
        {
            metrics = [accuracy, precision, recall, f1],
            show    = overfitting_analysis
        }
     """)),

    # ── V14 : sample_2.ayar inline (SVM + NaiveBayes + LR / spam) ────────────
    ("V14 – sample_2 inline: SVM + NaiveBayes + LogisticRegression, spam",
     open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "sample_2.ayar")).read()
     if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "sample_2.ayar")) else
     textwrap.dedent("""\
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
                threshold = 0.10;,
                underfit_max_acc = 0.70;
            }
            compare by recall lower_is_better;
            exclude if underfit;
            select best;
        }
        evaluate lr_spam on spam.test;
        report spam_report
        {
            metrics = [precision, recall, f1],
            show    = "confusion_matrix"
        }
     """)),

    # ── V15 : sample_3.ayar inline (all five models / churn) ─────────────────
    ("V15 – sample_3 inline: all five models, churn dataset (most complete)",
     open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "sample_3.ayar")).read()
     if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "sample_3.ayar")) else
     textwrap.dedent("""\
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
                threshold = 0.05;,
                underfit_max_acc = 0.60;
            }
            compare by accuracy higher_is_better;
            exclude if overfit or underfit;
            select best;
        }
        evaluate [churn_exp.best] on churn.test;
        report churn_report
        {
            metrics = [accuracy, precision, recall, f1],
            show    = overfitting_analysis
        }
     """)),
]


# ═════════════════════════════════════════════════════════════════════════════
# INVALID TEST CASES  (programs that MUST be rejected)
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is a (label, source, expected_fragment) tuple.
# expected_fragment is a substring that must appear in the error message.
# ═════════════════════════════════════════════════════════════════════════════

INVALID_TESTS = [

    # ── Grammar errors ────────────────────────────────────────────────────────

    # I1 : missing ';' after dataset declaration
    ("I1 – missing ';' after dataset declaration (ParseError)",
     textwrap.dedent("""\
        dataset titanic = load("data/titanic.csv")

        model KNN k
        {
            k = 3
        }
     """),
     "SEMICOLON"),

    # I2 : unknown model type 'RandomForest'
    ("I2 – unknown model type 'RandomForest' (ParseError)",
     textwrap.dedent("""\
        dataset bank = load("data/bank.csv");

        model RandomForest rf_model
        {
            n_estimators = 100,
            max_depth = 10
        }
     """),
     "model type"),

    # I3 : invalid split role 'verify' instead of 'validation'
    ("I3 – invalid split role 'verify' (ParseError)",
     textwrap.dedent("""\
        dataset wine = load("data/wine.csv");
        split wine into
            train(0.60),
            verify(0.20),
            test(0.20);
     """),
     "split role"),

    # I4 : missing '=' in model field  (identifier directly followed by value)
    ("I4 – missing '=' in model field (ParseError)",
     textwrap.dedent("""\
        model KNN k
        {
            neighbours 5
        }
     """),
     "OP_ASSIGN"),

    # I5 : unknown top-level keyword 'pipeline'
    ("I5 – unknown top-level keyword 'pipeline' (ParseError)",
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

    # I6 : missing 'into' keyword in split statement
    ("I6 – missing 'into' in split statement (ParseError)",
     "split ds train(0.7), validation(0.15), test(0.15);",
     "into"),

    # I7 : 'loss' is not a recognised metric literal
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

    # I8 : missing '.' in dataset reference (train k on d train)
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

    # I9 : unclosed experiment block (missing closing '}')
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

    # I10 : invalid compare direction 'max_is_better'
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

    # ── Layout errors ─────────────────────────────────────────────────────────

    # I11 : '{' on the same line as 'model KNN k'
    ("I11 – layout: '{' on same line as model header (LayoutError)",
     "model KNN k {\n    n = 5\n}",
     "LayoutError"),

    # I12 : '{' on the same line as 'experiment e'
    ("I12 – layout: '{' on same line as experiment header (LayoutError)",
     'dataset d = load("x.csv");\nexperiment e {\n    select best;\n}',
     "LayoutError"),

    # I13 : '{' on the same line as 'report r'
    ("I13 – layout: '{' on same line as report header (LayoutError)",
     "report r {\n    metrics = [accuracy],\n    show = overfitting_analysis\n}",
     "LayoutError"),

    # I14 : '{' on same line as 'analyze overfitting' (inside experiment)
    ("I14 – layout: '{' on same line as analyze overfitting (LayoutError)",
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

    # I15 : '}' on the same line as the last field content
    ("I15 – layout: '}' on same line as field content (LayoutError)",
     "model KNN k\n{\n    n = 5 }",
     "LayoutError"),
]


# ═════════════════════════════════════════════════════════════════════════════
# TEST RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

def run_valid_tests() -> tuple:
    """Run all VALID_TESTS and return (passed, failed) counts."""
    passed = failed = 0
    print(BOLD("\n── VALID tests (must parse without errors) " + "─" * 20))
    for label, source in VALID_TESTS:
        ok = run_valid_test(label, source)
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed


def run_invalid_tests() -> tuple:
    """Run all INVALID_TESTS and return (passed, failed) counts."""
    passed = failed = 0
    print(BOLD("\n── INVALID tests (must raise an error) " + "─" * 23))
    for label, source, fragment in INVALID_TESTS:
        ok = run_invalid_test(label, source, fragment)
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed


def run_builtin_suite():
    """Execute the full built-in test suite and print a summary."""
    print(BOLD("\n" + "═" * 64))
    print(BOLD("  ayar DSL – D3 Test Suite"))
    print(BOLD("  Imports: ayar_lexer.py, ayar_parser.py  (unmodified)"))
    print(BOLD("  Adapter: adapt_tokens()  fixes token-name mismatches"))
    print(BOLD("  Layout rule: '{' and '}' must each be on their own line"))
    print(BOLD("═" * 64))

    vp, vf = run_valid_tests()
    ip, if_ = run_invalid_tests()

    total_p = vp + ip
    total_f = vf + if_
    total   = total_p + total_f

    print(BOLD("\n" + "═" * 64))
    print(BOLD(f"  Valid   tests : {vp}/{vp + vf} passed"))
    print(BOLD(f"  Invalid tests : {ip}/{ip + if_} passed"))
    summary = f"  Total         : {total_p}/{total} passed"
    if total_f == 0:
        print(GREEN(summary))
    else:
        print(RED(summary + f"  ({total_f} FAILED)"))
    print(BOLD("═" * 64 + "\n"))


# ═════════════════════════════════════════════════════════════════════════════
# CLI UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def run_file(path: str):
    """Parse a .ayar file from disk and print its AST."""
    print(BOLD(f"\nParsing file: {path}"))
    try:
        source = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(RED(f"File not found: {path}"))
        sys.exit(1)
    try:
        tokens = Lexer(source).tokenize()
        tokens = adapt_tokens(tokens)
        ast    = Parser(tokens).parse()
        print(CYAN("\nAST:"))
        for line in Parser.dump_ast(ast).splitlines():
            print("  " + line)
        print(GREEN("\n✓ Parse succeeded."))
    except LexicalError as err:
        print(RED(f"\n✗ LexicalError: {err}"))
        sys.exit(1)
    except LayoutError as err:
        print(RED(f"\n✗ LayoutError: {err}"))
        sys.exit(1)
    except ParseError as err:
        print(RED(f"\n✗ ParseError: {err}"))
        sys.exit(1)


def run_tokens_only(path: str):
    """Lex a .ayar file and print the token stream (no parsing)."""
    try:
        source = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        print(RED(f"File not found: {path}"))
        sys.exit(1)
    tokens = Lexer(source).tokenize()
    tokens = adapt_tokens(tokens)     # adapter runs here too
    print(BOLD(f"\nToken stream for: {path}"))
    w_type  = max(len(t.type)  for t in tokens) + 2
    w_value = max(len(t.value) for t in tokens) + 2
    hdr = f"  {'TYPE':<{w_type}} {'VALUE':<{w_value}} {'LINE':>5}  {'COL':>5}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for tok in tokens:
        print(f"  {tok.type:<{w_type}} {tok.value!r:<{w_value}} "
              f"{tok.line_number:>5}  {tok.column_number:>5}")


def run_expr(expr_source: str):
    """Parse a single expression and print its AST."""
    print(BOLD(f"\nParsing expression: {expr_source!r}"))
    try:
        tokens = Lexer(expr_source).tokenize()
        tokens = adapt_tokens(tokens)
        ast    = Parser(tokens).parse_expr()
        print(CYAN("  AST:"))
        for line in Parser.dump_ast(ast).splitlines():
            print("  " + line)
        print(GREEN("  ✓ Parsed successfully."))
    except (LexicalError, LayoutError, ParseError) as err:
        print(RED(f"  ✗ Error: {err}"))


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Test runner for the ayar DSL lexer + parser (CSE341 D3)."
    )
    ap.add_argument("--file",   metavar="PATH",
                    help="Parse a .ayar file from disk and print its AST.")
    ap.add_argument("--tokens", metavar="PATH",
                    help="Show the raw token stream for a .ayar file.")
    ap.add_argument("--expr",   metavar="EXPR",
                    help='Parse a single expression, e.g. --expr "3+4*2".')
    args = ap.parse_args()

    if   args.tokens: run_tokens_only(args.tokens)
    elif args.file:   run_file(args.file)
    elif args.expr:   run_expr(args.expr)
    else:             run_builtin_suite()


if __name__ == "__main__":
    main()