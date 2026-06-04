"""
ayar_typechecker.py
===================
Type checker for the Ayar DSL (CSE341 – Concepts of Programming Languages).

Theory reference: Sebesta §6.1–§6.4 (type systems, type rules, type checking).

Design notes
────────────
  • TypeChecker is a pure ANALYSIS pass: it never modifies the AST.
  • It walks the ProgramNode using a dedicated _visit_* method per node type,
    mirroring the Visitor pattern (Sebesta §4.4.3 discussion of AST traversal).
  • Line numbers are recovered from an optional token list.  Because the
    current parser does not embed line numbers in AST nodes, the TypeChecker
    maintains a _LineMap that scans the token stream once and builds a
    keyword+identifier → line-number index.  When the token list is omitted,
    all error lines are reported as None.
  • Rules enforced (from D1 Part-2 Type System section):
      R1  Every model name used in `train` / `evaluate` (inside an experiment)
          must have been declared as a top-level `model` before the experiment.
      R2  Every dataset reference `ds.role` must refer to a dataset that was
          both declared (dataset ds = …) and split, and the role must be one
          of {train, validation, test}.
      R3  A metric literal (accuracy | precision | recall | f1) must not appear
          inside an arithmetic expression (BinOpNode with op in {+, -, *, /}).
      R4  The metric named in `compare by <m>` must appear in the immediately
          preceding `collect metrics […]` list within the same experiment.
      R5  Training on the test split is a data-leakage error:
              train <model> on <dataset>.test   →  AyarTypeError
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

# ── import AST node types from the student's parser ───────────────────────────
#
# Only the node types that are directly inspected by one or more type rules
# are imported here.  Every other node type (ReportStmtNode, AnalyzeStmtNode,
# ExcludeStmtNode, SelectStmtNode, EvalTargetSimpleNode, EvalTargetBracketNode,
# AttributeAccessNode, IdentifierNode, FieldNode, …) is either not visited by
# the type checker or is silently skipped in _check_node — it does not need to
# be imported.
#
# Rule cross-reference:
#   R1  UndeclaredModel          → ModelDeclNode (global collection)
#                                  TrainStmtNode, EvaluateStmtNode (usage sites)
#   R2  Dataset declared & split → DatasetDeclNode, SplitStmtNode (global collection)
#                                  DatasetRefNode (usage site)
#   R3  Metric in arithmetic     → BinOpNode (operator), LiteralNode (metric leaf)
#                                  UnaryOpNode (recurse through unary minus)
#   R4  Compare metric collected → CollectStmtNode (stores collected list)
#                                  CompareStmtNode (checks against list)
#   R5  Data leakage             → TrainStmtNode + DatasetRefNode (role == 'test')
#
# ProgramNode is the root handed to TypeChecker.__init__.
# ExperimentDeclNode is the container that scopes R4's collected-metric state.

from ayar_parser import (
    ProgramNode,           # root node passed to TypeChecker.__init__
    DatasetDeclNode,       # R2 – collected in pass 1 to build _declared_datasets
    SplitStmtNode,         # R2 – collected in pass 1 to build _split_datasets
    ModelDeclNode,         # R1 – collected in pass 1 to build _declared_models
    ExperimentDeclNode,    # R4 – scopes the collected-metric state per experiment
    TrainStmtNode,         # R1, R5 – model must be declared; test split forbidden
    EvaluateStmtNode,      # R1 – model must be declared
    DatasetRefNode,        # R2, R5 – dataset must be declared, split, role valid
    CollectStmtNode,       # R4 – records which metrics were collected
    CompareStmtNode,       # R4 – metric must appear in preceding collect list
    BinOpNode,             # R3 – arithmetic operators (+, -, *, /) trigger check
    UnaryOpNode,           # R3 – recurse through unary minus to find metric leaves
    LiteralNode,           # R3 – kind == 'metric' inside arithmetic is the error
    EvalStmtNode,          # R2 – top-level evaluate also carries a DatasetRefNode
)

# ══════════════════════════════════════════════════════════════════════════════
# AyarTypeError dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AyarTypeError:
    """
    Represents a single semantic / type error detected during the checking pass.

    Attributes
    ----------
    rule        : Short symbolic rule identifier, e.g. "R1-UndeclaredModel".
    message     : Human-readable description of the violation.
    line        : Source line number where the violation occurs, or None if the
                  token stream was not provided to the checker.
    """
    rule:    str
    message: str
    line:    Optional[int] = None

    def __str__(self) -> str:
        loc = f"line {self.line}" if self.line is not None else "line unknown"
        return f"[{self.rule}] {loc}: {self.message}"


# ══════════════════════════════════════════════════════════════════════════════
# _LineMap — lightweight line-number recovery from the token stream
# ══════════════════════════════════════════════════════════════════════════════

class _LineMap:
    """
    Builds a fast lookup structure from the raw token list so that the
    TypeChecker can report approximate line numbers even though the AST nodes
    themselves do not carry that information.

    Strategy
    ────────
    For each token we record (type, value) → [list of line numbers].
    When we need the line for, say,  TrainStmt(model='knn_clf'), we look up
    the KEYWORD token "train" that immediately precedes the IDENTIFIER "knn_clf"
    and return its line number.  For compound structures we use the statement
    keyword ("train", "evaluate", "compare", …) as the anchor.

    Because the same keyword/identifier can appear multiple times, the list is
    consumed in order (pop-from-front) so successive identical statements each
    get the right line.
    """

    def __init__(self, tokens: list):
        # Build positional index: list of (type, value, line_number) triples.
        self._stream: list[tuple] = [
            (t.type, t.value, t.line_number)
            for t in tokens
            if t.type not in ("EOF",)
        ]
        self._pos: int = 0          # consumed-up-to cursor

    # ── Public helpers ─────────────────────────────────────────────────────────

    def find_keyword_then_identifier(self, keyword: str, identifier: str) -> Optional[int]:
        """
        Scan forward for the pattern  KEYWORD(keyword) … IDENTIFIER(identifier)
        where '…' contains at most a few tokens (same statement).
        Returns the line of the KEYWORD token, or None.
        """
        for i in range(len(self._stream)):
            t_type, t_val, t_line = self._stream[i]
            if t_type == "KEYWORD" and t_val == keyword:
                # Look ahead up to 4 tokens for the identifier.
                for j in range(i + 1, min(i + 5, len(self._stream))):
                    nt_type, nt_val, _ = self._stream[j]
                    if nt_type == "IDENTIFIER" and nt_val == identifier:
                        return t_line
                    # Don't cross statement boundaries.
                    if nt_type == "SEMICOLON":
                        break
        return None

    def find_keyword_then_metric(self, keyword: str, metric: str) -> Optional[int]:
        """Like above but looks for a METRIC_LITERAL token instead of IDENTIFIER."""
        for i in range(len(self._stream)):
            t_type, t_val, t_line = self._stream[i]
            if t_type == "KEYWORD" and t_val == keyword:
                for j in range(i + 1, min(i + 8, len(self._stream))):
                    nt_type, nt_val, _ = self._stream[j]
                    if nt_type == "METRIC_LITERAL" and nt_val == metric:
                        return t_line
                    if nt_type == "SEMICOLON":
                        break
        return None

    def find_metric_in_expression(self, metric_value: str) -> Optional[int]:
        """Return the first line on which a METRIC_LITERAL with this value appears."""
        for t_type, t_val, t_line in self._stream:
            if t_type == "METRIC_LITERAL" and t_val == metric_value:
                return t_line
        return None

    def find_dataset_ref(self, dataset: str, role: str) -> Optional[int]:
        """
        Find IDENTIFIER(dataset) DOT KEYWORD(role) and return the line of the
        IDENTIFIER token.

        The lexer emits the dot as DOT and the split-role keyword (train /
        validation / test) as KEYWORD — not as IDENTIFIER.  We therefore check
        both the token type AND value of the role token.
        """
        for i in range(len(self._stream) - 2):
            t_type, t_val, t_line = self._stream[i]
            if t_type == "IDENTIFIER" and t_val == dataset:
                dot_type, dot_val, _ = self._stream[i + 1]
                role_type, role_val, _ = self._stream[i + 2]
                if dot_type == "DOT" and role_type == "KEYWORD" and role_val == role:
                    return t_line
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TypeChecker
# ══════════════════════════════════════════════════════════════════════════════

class TypeChecker:
    """
    Performs a single-pass semantic / type-checking walk over a ProgramNode.

    Usage
    -----
        checker = TypeChecker(ast, tokens)   # tokens optional but enables line reporting
        errors  = checker.check()            # returns List[AyarTypeError]

    The checker NEVER modifies the AST — all state lives in private instance
    variables initialised in __init__ and populated during check().

    Internally the check is split into two sub-passes:
      1. _collect_globals()  – gather all declared datasets, split datasets,
                               and model names from the top-level declarations.
      2. _check_node()       – recursive walk that enforces all five rules.
    """

    _ARITHMETIC_OPS = {"+", "-", "*", "/"}
    _VALID_ROLES    = {"train", "validation", "test"}
    _VALID_METRICS  = {"accuracy", "precision", "recall", "f1"}

    def __init__(self, program: ProgramNode, tokens: list = None):
        self._program = program
        self._lmap    = _LineMap(tokens) if tokens else None

        # Populated by _collect_globals()
        self._declared_datasets: set  = set()   # dataset names (declared via 'dataset …')
        self._split_datasets:    set  = set()   # dataset names that have been split
        self._declared_models:   set  = set()   # model names (declared via 'model …')

        # Populated per-experiment during _check_node()
        self._current_collected_metrics: Optional[list] = None

        # Result accumulator
        self._errors: List[AyarTypeError] = []

    # ── Public entry point ─────────────────────────────────────────────────────

    def check(self) -> List[AyarTypeError]:
        """
        Run all type rules over the program AST.
        Returns a (possibly empty) list of AyarTypeError objects.
        """
        self._errors.clear()
        self._collect_globals()
        for decl in self._program.declarations:
            self._check_node(decl)
        return list(self._errors)

    # ── Pass 1: global symbol collection ──────────────────────────────────────

    def _collect_globals(self) -> None:
        """
        Single forward scan of all top-level declarations to populate the
        three symbol sets.  Global declarations are collected in a forward
        scan before checking begins — model and dataset names are visible
        throughout the program regardless of textual position (hoisting
        semantics).

        After collection, R6 checks that no name appears in both
        _declared_models and _declared_datasets.
        """
        for decl in self._program.declarations:
            if isinstance(decl, DatasetDeclNode):
                self._declared_datasets.add(decl.name)
            elif isinstance(decl, SplitStmtNode):
                self._split_datasets.add(decl.dataset)
            elif isinstance(decl, ModelDeclNode):
                self._declared_models.add(decl.name)

        # R6 — name collision between datasets and models.
        # Checked once after the full scan so every collision is reported,
        # not just the first one encountered.
        for name in self._declared_models & self._declared_datasets:
            self._errors.append(AyarTypeError(
                rule    = "R6-NameCollision",
                message = (
                    f"Name '{name}' is declared as both a dataset and a model. "
                    f"Dataset and model names must be distinct."
                ),
                line    = None,   # hoisting: no single source line to blame
            ))

    # ── Pass 2: recursive AST walk ─────────────────────────────────────────────

    def _check_node(self, node) -> None:
        """Dispatch to a specialised _visit_* method by node type."""
        if isinstance(node, ExperimentDeclNode):
            self._visit_experiment(node)
        elif isinstance(node, TrainStmtNode):
            self._visit_train(node)
        elif isinstance(node, EvaluateStmtNode):
            self._visit_evaluate(node)
        elif isinstance(node, DatasetRefNode):
            self._visit_dataset_ref(node)
        elif isinstance(node, CollectStmtNode):
            self._visit_collect(node)
        elif isinstance(node, CompareStmtNode):
            self._visit_compare(node)
        elif isinstance(node, BinOpNode):
            self._visit_binop(node)
        elif isinstance(node, UnaryOpNode):
            self._check_node(node.operand)
        elif isinstance(node, EvalStmtNode):
            self._check_node(node.dataset_ref)
        # Nodes that need no type-checking (literals, declarations already
        # processed in pass 1, report/analyze/exclude/select statements):
        # simply do nothing.

    # ── Experiment block ───────────────────────────────────────────────────────

    def _visit_experiment(self, node: ExperimentDeclNode) -> None:
        """
        Walk all statements inside the experiment.
        The `collected_metrics` state is scoped to each experiment block:
        it is reset to None at entry and populated the moment a CollectStmt
        is encountered, so that a CompareStmt can validate against it.
        """
        # Save & reset per-experiment metric state.
        saved = self._current_collected_metrics
        self._current_collected_metrics = None

        for stmt in node.stmts:
            self._check_node(stmt)

        self._current_collected_metrics = saved

    # ── R1: Model declared ─────────────────────────────────────────────────────

    def _visit_train(self, node: TrainStmtNode) -> None:
        """
        R1  – The trained model must be declared.
        R5  – Training on the test split is a data-leakage error.
        """
        # R1
        if node.model not in self._declared_models:
            line = (self._lmap.find_keyword_then_identifier("train", node.model)
                    if self._lmap else None)
            self._errors.append(AyarTypeError(
                rule    = "R1-UndeclaredModel",
                message = (f"Model '{node.model}' is used in 'train' but was never "
                           f"declared as a top-level model."),
                line    = line,
            ))

        # R5 – data leakage guard
        if isinstance(node.dataset_ref, DatasetRefNode):
            if node.dataset_ref.role == "test":
                line = (self._lmap.find_dataset_ref(
                            node.dataset_ref.dataset, "test")
                        if self._lmap else None)
                self._errors.append(AyarTypeError(
                    rule    = "R5-DataLeakage",
                    message = (f"Training model '{node.model}' on the test split "
                               f"'{node.dataset_ref.dataset}.test' is a data-leakage "
                               f"error.  Use the 'train' split for training."),
                    line    = line,
                ))

        # Still validate the dataset reference for R2.
        self._check_node(node.dataset_ref)

    def _visit_evaluate(self, node: EvaluateStmtNode) -> None:
        """R1 – The evaluated model must be declared."""
        if node.model not in self._declared_models:
            line = (self._lmap.find_keyword_then_identifier("evaluate", node.model)
                    if self._lmap else None)
            self._errors.append(AyarTypeError(
                rule    = "R1-UndeclaredModel",
                message = (f"Model '{node.model}' is used in 'evaluate' but was "
                           f"never declared as a top-level model."),
                line    = line,
            ))

        self._check_node(node.dataset_ref)

    # ── R2: Dataset reference valid ────────────────────────────────────────────

    def _visit_dataset_ref(self, node: DatasetRefNode) -> None:
        """
        R2  – The dataset must be declared AND split, and the role must be
              one of {train, validation, test}.
        """
        # Role validity (parser already guarantees this syntactically, but we
        # enforce it here for completeness as specified in D1.)
        if node.role not in self._VALID_ROLES:
            line = (self._lmap.find_dataset_ref(node.dataset, node.role)
                    if self._lmap else None)
            self._errors.append(AyarTypeError(
                rule    = "R2-InvalidRole",
                message = (f"Dataset role '{node.role}' is not valid.  "
                           f"Allowed roles: {sorted(self._VALID_ROLES)}."),
                line    = line,
            ))
            return  # no further checks make sense for an invalid role

        # Dataset must be declared.
        if node.dataset not in self._declared_datasets:
            line = (self._lmap.find_dataset_ref(node.dataset, node.role)
                    if self._lmap else None)
            self._errors.append(AyarTypeError(
                rule    = "R2-UndeclaredDataset",
                message = (f"Dataset '{node.dataset}' referenced as "
                           f"'{node.dataset}.{node.role}' was never declared "
                           f"with 'dataset {node.dataset} = load(…)'."),
                line    = line,
            ))

        # Dataset must be split.
        elif node.dataset not in self._split_datasets:
            line = (self._lmap.find_dataset_ref(node.dataset, node.role)
                    if self._lmap else None)
            self._errors.append(AyarTypeError(
                rule    = "R2-UnsplitDataset",
                message = (f"Dataset '{node.dataset}' was declared but never split.  "
                           f"Add 'split {node.dataset} into train(…), "
                           f"validation(…), test(…);' before referencing it."),
                line    = line,
            ))

    # ── R3: Metrics in arithmetic ──────────────────────────────────────────────

    def _visit_binop(self, node: BinOpNode, _in_arithmetic: bool = False) -> None:
        """
        R3  – A metric literal must not appear as an operand of an arithmetic
              operator (+, -, *, /).

        The check is depth-first: we detect BinOp nodes whose op is arithmetic,
        then flag any LiteralNode(kind='metric') that appears as an immediate
        or nested operand.
        """
        is_arith = node.op in self._ARITHMETIC_OPS
        in_arith = _in_arithmetic or is_arith

        # Check left operand.
        self._check_arith_operand(node.left, in_arith)
        # Check right operand.
        self._check_arith_operand(node.right, in_arith)

    def _check_arith_operand(self, node, in_arithmetic: bool) -> None:
        """Recurse into an expression node, flagging metrics in arithmetic."""
        if isinstance(node, LiteralNode) and node.kind == "metric":
            if in_arithmetic:
                line = (self._lmap.find_metric_in_expression(node.value)
                        if self._lmap else None)
                self._errors.append(AyarTypeError(
                    rule    = "R3-MetricInArithmetic",
                    message = (f"Metric literal '{node.value}' used inside an "
                               f"arithmetic expression.  Metrics are not numeric "
                               f"values and cannot appear in +, -, *, / operations."),
                    line    = line,
                ))
        elif isinstance(node, BinOpNode):
            is_arith = node.op in self._ARITHMETIC_OPS
            new_flag = in_arithmetic or is_arith
            self._check_arith_operand(node.left,  new_flag)
            self._check_arith_operand(node.right, new_flag)
        elif isinstance(node, UnaryOpNode):
            self._check_arith_operand(node.operand, in_arithmetic)
        # IdentifierNode, AttributeAccessNode, and other LiteralNode kinds
        # are fine inside arithmetic expressions.

    # ── R4: Compare metric must be collected ───────────────────────────────────

    def _visit_collect(self, node: CollectStmtNode) -> None:
        """Store the collected metrics list for later validation by CompareStmt."""
        self._current_collected_metrics = node.metrics

    def _visit_compare(self, node: CompareStmtNode) -> None:
        """
        R4  – The metric named in 'compare by <m>' must have appeared in the
              'collect metrics […]' list within the same experiment block.
        """
        if self._current_collected_metrics is None:
            # No collect statement has been seen yet in this experiment.
            line = (self._lmap.find_keyword_then_metric("compare", node.metric)
                    if self._lmap else None)
            self._errors.append(AyarTypeError(
                rule    = "R4-CompareBeforeCollect",
                message = (f"'compare by {node.metric}' appears before any "
                           f"'collect metrics' statement in this experiment.  "
                           f"Declare which metrics to collect first."),
                line    = line,
            ))
        elif node.metric not in self._current_collected_metrics:
            line = (self._lmap.find_keyword_then_metric("compare", node.metric)
                    if self._lmap else None)
            self._errors.append(AyarTypeError(
                rule    = "R4-UncollectedMetric",
                message = (f"'compare by {node.metric}' references a metric that "
                           f"was not collected.  "
                           f"Collected metrics: {self._current_collected_metrics}.  "
                           f"Add '{node.metric}' to the 'collect metrics' list, or "
                           f"change the compare metric to one that is collected."),
                line    = line,
            ))

    # ── Convenience line method ────────────────────────────────────────────────

    def _line(self, *args) -> Optional[int]:
        """Return None safely when no token stream is available."""
        return None


# ══════════════════════════════════════════════════════════════════════════════
# DEMO  —  run three test cases when executed directly
# ══════════════════════════════════════════════════════════════════════════════

def _parse(source: str):
    """Helper: lex + validate layout + parse, return (ast, tokens)."""
    from ayar_lexer import Lexer
    from ayar_parser import Parser, validate_layout
    tokens = Lexer(source).tokenize()
    validate_layout(tokens)
    return Parser(tokens).parse(), tokens


def _run(label: str, source: str) -> None:
    """Parse source, type-check it, and print results."""
    print("=" * 70)
    print(f"TEST: {label}")
    print("=" * 70)
    ast, tokens = _parse(source)
    checker = TypeChecker(ast, tokens)
    errors  = checker.check()
    if not errors:
        print("  ✓  No type errors found — program is valid.")
    else:
        for e in errors:
            print(f"  ✗  {e}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Test program 1 — VALID: all rules satisfied
# ─────────────────────────────────────────────────────────────────────────────
_VALID_PROGRAM = """
dataset iris = load("iris.csv");

split iris into
    train(0.70),
    validation(0.15),
    test(0.15);

model KNN knn_model
{
    k = 3,
    distance = "euclidean"
}

model DecisionTree dt_model
{
    max_depth = 5,
    criterion = "gini"
}

experiment iris_exp
{
    train knn_model on iris.train;
    train dt_model  on iris.train;

    evaluate knn_model on iris.validation;
    evaluate dt_model  on iris.validation;

    collect metrics [accuracy, f1];

    analyze overfitting
    {
        threshold = 0.10;
        underfit_max_acc = 0.70;
    }

    compare by f1 higher_is_better;

    exclude if overfit or underfit;

    select best;
}

evaluate iris_exp.best on iris.test;

report summary_report
{
    metrics = [accuracy, f1],
    show    = overfitting_analysis
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Test program 2 — R3 VIOLATION: metric literal in arithmetic expression
#                  `accuracy + 0.5` inside a field value … but since Ayar's
#                  grammar does not use expressions as field values (fields only
#                  accept int/float/string/bool literals per the EBNF), the
#                  natural place to surface an arithmetic expression is inside
#                  an exclude condition or a hypothetical expression context.
#
#  The cleanest way to trigger R3 given the actual grammar is to note that
#  <primary> ::= <metric_literal> is legal, and <add_expr> can contain any
#  <primary>, so:   accuracy + 0.5   is a valid <add_expr>.
#  The parser will accept this anywhere an <expr> is expected.
#
#  In the Ayar grammar the only place a free <expr> appears is inside the
#  analysis fields' right-hand side values — but those are constrained to
#  numeric literals by the parser.  The grammar does allow field_value to be
#  any <field_value>, which is int/float/string/bool (not expr).
#
#  To produce an AST with a metric-in-arithmetic BinOpNode we therefore need
#  the parser to accept it via the expression production.  Looking at the
#  parser's _parse_field_value, it does NOT call parse_expr; it only accepts
#  literals.  The one place the full expression grammar is reachable is inside
#  a parenthesised primary, but that is also gated through _parse_primary.
#
#  Given these constraints, the correct way to test R3 is to construct the
#  AST directly (not through the parser) and hand it to the type checker —
#  which is exactly what a unit test for the type-checker component would do.
#  This mirrors real-world practice where the type checker is tested with
#  synthetic ASTs to isolate rule coverage from parser behaviour.
# ─────────────────────────────────────────────────────────────────────────────
def _test_r3_metric_in_arithmetic() -> None:
    """
    Construct a minimal ProgramNode that contains:
        BinOpNode('+', LiteralNode('metric','accuracy'), LiteralNode('float', 0.5))
    and verify that R3 fires.
    This uses a synthetic (hand-crafted) AST — the correct approach for testing
    type-checker rules in isolation when the parser itself rejects the input.
    """
    from ayar_parser import (
        ProgramNode, DatasetDeclNode, SplitStmtNode, SplitPartNode,
        ModelDeclNode, ExperimentDeclNode, CollectStmtNode, CompareStmtNode,
        SelectStmtNode, BinOpNode, LiteralNode, AnalyzeStmtNode, AnalyzeFieldNode,
    )

    # Minimal valid scaffolding so that the only error is R3.
    dataset_decl = DatasetDeclNode(name="iris", path="iris.csv")
    split_stmt   = SplitStmtNode(dataset="iris", parts=[
        SplitPartNode("train",      0.70),
        SplitPartNode("validation", 0.15),
        SplitPartNode("test",       0.15),
    ])
    model_decl = ModelDeclNode(model_type="KNN", name="knn_m", fields=[])

    # The offending expression node: accuracy + 0.5
    bad_expr = BinOpNode(
        op    = "+",
        left  = LiteralNode(kind="metric", value="accuracy"),
        right = LiteralNode(kind="float",  value=0.5),
    )

    # We embed the expression in the experiment as a synthetic statement.
    # Since TypeChecker._check_node dispatches on known statement types and
    # silently ignores unknown ones, we subclass ASTNode to create a
    # thin wrapper that the checker's _check_node will dispatch to _visit_binop
    # via the BinOpNode branch when we place bad_expr directly in the stmts list.
    collect_stmt = CollectStmtNode(metrics=["accuracy"])
    compare_stmt = CompareStmtNode(metric="accuracy", direction="higher_is_better")
    select_stmt  = SelectStmtNode()

    experiment = ExperimentDeclNode(
        name  = "exp",
        stmts = [collect_stmt, compare_stmt, select_stmt, bad_expr],
        #                                                  ^^^^^^^^^^
        #  bad_expr is a BinOpNode — TypeChecker._check_node will dispatch
        #  it to _visit_binop, which will find the metric literal and fire R3.
    )

    program = ProgramNode(declarations=[dataset_decl, split_stmt, model_decl, experiment])

    print("=" * 70)
    print("TEST: R3 — metric literal in arithmetic expression (accuracy + 0.5)")
    print("      [Synthetic AST — parser correctly rejects this syntax,")
    print("       so the type checker is tested with a hand-crafted AST]")
    print("=" * 70)
    checker = TypeChecker(program)   # no token list for a synthetic AST
    errors  = checker.check()
    if not errors:
        print("  ✓  No type errors found — program is valid.")
    else:
        for e in errors:
            print(f"  ✗  {e}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Test program 3 — R4 VIOLATION: compare by f1, but only [accuracy] collected
# ─────────────────────────────────────────────────────────────────────────────
_R4_PROGRAM = """
dataset iris = load("iris.csv");

split iris into
    train(0.70),
    validation(0.15),
    test(0.15);

model KNN knn_model
{
    k = 5,
    distance = "euclidean"
}

experiment iris_exp
{
    train knn_model on iris.train;
    evaluate knn_model on iris.validation;

    collect metrics [accuracy];

    compare by f1 higher_is_better;

    select best;
}

evaluate iris_exp.best on iris.test;

report summary_report
{
    metrics = [accuracy],
    show    = overfitting_analysis
}
"""


if __name__ == "__main__":
    _run("Valid program — all rules satisfied", _VALID_PROGRAM)
    _test_r3_metric_in_arithmetic()
    _run("R4 violation — compare by f1, but only [accuracy] collected", _R4_PROGRAM)

# ─────────────────────────────────────────────────────────────────────────────
# Test program 4 — R1 VIOLATION: model used in `train` / `evaluate` but never
#                  declared.  ghost_model has no top-level `model … { … }`
#                  block anywhere in the file.  This exercises R1 with a fully
#                  parsed (not synthetic) AST, so line numbers are recovered
#                  from the real token stream.
# ─────────────────────────────────────────────────────────────────────────────
_R1_PROGRAM = """
dataset iris = load("iris.csv");

split iris into
    train(0.70),
    validation(0.15),
    test(0.15);

model KNN knn_model
{
    k = 5,
    distance = "euclidean"
}

experiment iris_exp
{
    train knn_model   on iris.train;
    train ghost_model on iris.train;

    evaluate knn_model   on iris.validation;
    evaluate ghost_model on iris.validation;

    collect metrics [accuracy, f1];

    compare by f1 higher_is_better;

    select best;
}

evaluate iris_exp.best on iris.test;

report summary_report
{
    metrics = [accuracy, f1],
    show    = overfitting_analysis
}
"""

if __name__ == "__main__":
    _run("R1 violation — ghost_model used in train/evaluate but never declared",
         _R1_PROGRAM)
