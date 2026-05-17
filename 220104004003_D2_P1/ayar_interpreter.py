"""
ayar_interpreter.py
===================
Interpreter for the Ayar DSL (CSE341 – Concepts of Programming Languages).

Theory reference: Sebesta, "Concepts of Programming Languages" 12th ed.
  – §3.5   : Operational Semantics (state-transition rules)
  – §5.4   : Binding and Binding Time
  – §5.7   : Static Scoping
  – §5.8   : Lifetime (stack-dynamic experiment scope)
  – §6.1   : Type System overview

Design
──────
The interpreter faithfully implements the formal state triple from D1 §4.4:

    σ = (Γ, M, R)

  Γ  (gamma)  – dict  : global environment, maps dataset and model names to
                        their declared record objects.  All top-level 'dataset'
                        and 'model' declarations write here.  The experiment
                        block also exports  exp_name.best  into Γ after it
                        concludes.

  M           – dict  : training / evaluation results for the CURRENT
                        experiment.  Created fresh (empty) each time an
                        experiment block starts; discarded when it ends.
                        Maps model name → ModelResult (metric values + tags).

  R           – str | None : the best model name selected by 'select best',
                        or None (⊥) if not yet selected.

Simulation (Option C) — one formula per model type
────────────────────────────────────────────────────
Each model type has a semantically motivated formula whose hyperparameter
values directly affect output metrics in a realistic direction.
Train split: accuracy += 0.07 to simulate training-set overfit.
Eval split:  base formula values used as-is.

Entry point
───────────
    Interpreter(ast, tokens).run()

The interpreter calls TypeChecker first and aborts on any AyarTypeError.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── Import the full AST node hierarchy from the parser ────────────────────────
from ayar_parser import (
    ProgramNode,
    DatasetDeclNode,
    SplitStmtNode,
    ModelDeclNode,
    FieldNode,
    LiteralNode,
    ExperimentDeclNode,
    TrainStmtNode,
    EvaluateStmtNode,
    DatasetRefNode,
    CollectStmtNode,
    AnalyzeStmtNode,
    AnalyzeFieldNode,
    CompareStmtNode,
    ExcludeStmtNode,
    SelectStmtNode,
    EvalStmtNode,
    EvalTargetSimpleNode,
    EvalTargetBracketNode,
    ReportStmtNode,
    ReportFieldMetricsNode,
    ReportFieldShowNode,
)

# ── Import the type checker ────────────────────────────────────────────────────
from ayar_typechecker import TypeChecker, AyarTypeError


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME ERROR TYPES
# ══════════════════════════════════════════════════════════════════════════════

class AyarRuntimeError(Exception):
    """Base class for all Ayar interpreter runtime errors."""


class DataLeakageError(AyarRuntimeError):
    """
    Raised when 'train model on dataset.test' is attempted.
    Rule E-Train, data leakage guard: dataset_ref.role ≠ test.
    The type checker (R5) catches this statically, but the interpreter
    also guards at runtime per the D1 semantics specification.
    """
    def __init__(self, model_name: str, dataset: str, line: Optional[int] = None):
        loc = f" (near line {line})" if line else ""
        msg = (
            f"\n[DataLeakageError]{loc}\n"
            f"  Model '{model_name}' was trained on the test split "
            f"'{dataset}.test'.\n"
            f"  Rule E-Train: dataset_ref.role ≠ test.\n"
            f"  Training on the test split exposes future data to the model\n"
            f"  and invalidates all performance estimates.  Use iris.train."
        )
        super().__init__(msg)


class NoEligibleModelError(AyarRuntimeError):
    """
    Raised by 'select best' when all models have been excluded (M is empty).
    Rule P-SelectBest (failure case): dom(M') = ∅  →  NoEligibleModelError.
    """
    def __init__(self, excluded_summary: str = ""):
        msg = (
            "\n[NoEligibleModelError]\n"
            "  'select best' was called but all models were excluded by the\n"
            "  preceding 'exclude if' statement.\n"
        )
        if excluded_summary:
            msg += f"  Excluded models:\n{excluded_summary}\n"
        msg += (
            "  To fix: relax the 'analyze overfitting' thresholds, or add\n"
            "  more models so that at least one survives the exclusion filter."
        )
        super().__init__(msg)


class UndefinedModelError(AyarRuntimeError):
    """
    Raised when a model name is referenced but was not bound in Γ.
    The type checker (R1) catches this statically; this is the runtime guard.
    """
    def __init__(self, name: str):
        super().__init__(
            f"[UndefinedModelError] Model '{name}' is not defined in Γ.\n"
            f"  Declare it with:  model <TYPE> {name} {{ ... }}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT RECORD TYPES  (values stored in Γ)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetRecord:
    """
    Bound into Γ by 'dataset id = load(path);'.
    Rule E-DatasetDecl.
    """
    name: str
    path: str
    # Split ratios, populated by E-SplitStmt.
    splits: Dict[str, float] = field(default_factory=dict)

    def is_split(self) -> bool:
        return bool(self.splits)


@dataclass
class ModelRecord:
    """
    Bound into Γ by 'model TYPE name { fields };'.
    Stores the declared hyperparameter values as a flat dict for
    easy lookup inside simulate().
    """
    model_type: str          # "KNN" | "DecisionTree" | "SVM" | "NaiveBayes" | "LogisticRegression"
    name:       str
    hyperparams: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# M — TRAINING RESULT TYPE  (values stored in M per-experiment)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    """
    Entry in M: the metric values and overfitting tags for one model.

    train_accuracy   – accuracy from the train-split run (+0.07 bonus).
                       Kept separately so analyze_overfitting can compute
                       train_acc - val_acc.
    eval_accuracy    – accuracy from the most recent evaluate-split run,
                       without the train bonus.  Stored separately from
                       'metrics' so that analyze_overfitting can compute
                       the gap even after 'collect' has restricted 'metrics'
                       to a list that does not include 'accuracy'.
    metrics          – MetricMap: the metrics currently in scope (filtered
                       by 'collect metrics').  Keys ⊆ {accuracy, precision,
                       recall, f1}.
    overfit          – Boolean tag set by analyze_overfitting.
    underfit         – Boolean tag set by analyze_overfitting.
    rank_key         – Numeric sort key set by compare (higher = better rank
                       when higher_is_better; lower = better when
                       lower_is_better).  None until compare runs.
    """
    model_name:     str
    model_type:     str
    train_accuracy: float                      # used by overfitting analysis
    eval_accuracy:  float = 0.0               # preserved even after collect
    metrics:        Dict[str, float] = field(default_factory=dict)
    overfit:        bool = False
    underfit:       bool = False
    rank_key:       Optional[float] = None     # populated by compare


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATE  — Option C: model-type formulae
# ══════════════════════════════════════════════════════════════════════════════

# Training-split accuracy bonus: simulates that a model always fits better on
# data it was trained on (overfitting signal).
_TRAIN_BONUS = 0.07


def simulate(model: ModelRecord, is_train_split: bool) -> Dict[str, float]:
    """
    Produce a deterministic MetricMap for the given model configuration.
    Returns all four metrics: accuracy, precision, recall, f1.

    Option C — each model type's formula uses its own semantically
    meaningful hyperparameters:

      KNN             : k controls the neighbourhood size.
                        Higher k → smoother boundaries → slightly lower
                        training accuracy but generally better generalisation
                        up to a plateau.
      DecisionTree    : max_depth controls how far the tree can grow.
                        Deeper → better fit, but hard-caps at 0.92 to model
                        the overfitting ceiling.
      SVM             : C is the regularisation inverse; higher C lets the
                        SVM fit training data more tightly.
      NaiveBayes      : No hyperparameter has a meaningful effect on overall
                        accuracy; fixed values are returned.
      LogisticRegression : max_iter controls convergence; more iterations →
                        better-fitted solution up to a plateau.

    The train-split run adds _TRAIN_BONUS to accuracy so that
    analyze_overfitting produces meaningful  (train_acc - val_acc) gaps.
    """
    hp = model.hyperparams
    mt = model.model_type

    if mt == "KNN":
        # k is usually 'k' or 'n_neighbors'; accept both field names.
        k = float(hp.get("k", hp.get("n_neighbors", 5)))
        # Higher k → accuracy decreases slightly (less memorisation).
        acc = max(0.60, 0.94 - k * 0.02)

    elif mt == "DecisionTree":
        max_depth = float(hp.get("max_depth", 5))
        # Deeper tree → better fit, capped at 0.92.
        acc = min(0.60 + max_depth * 0.05, 0.92)

    elif mt == "SVM":
        C = float(hp.get("C", 1))
        # Higher C → tighter fit, capped at 0.94.
        acc = min(0.70 + C * 0.03, 0.94)

    elif mt == "NaiveBayes":
        # var_smoothing affects log-probability stability, not accuracy much.
        acc = 0.78

    elif mt == "LogisticRegression":
        max_iter = float(hp.get("max_iter", 100))
        # More iterations → better convergence, capped at 0.91.
        acc = min(0.72 + max_iter * 0.0002, 0.91)

    else:
        # Unknown model type — should have been caught by the parser.
        acc = 0.70

    # Train-split bonus: simulates that the model has been fit to this data.
    if is_train_split:
        acc += _TRAIN_BONUS

    # Clip to [0.0, 1.0].
    acc = min(1.0, max(0.0, acc))

    # Derive the other three metrics from accuracy using model-specific offsets.
    # The offsets are chosen to give realistic spread across metrics.
    if mt == "KNN":
        f1        = acc - 0.01
        precision = acc + 0.01
        recall    = acc - 0.02

    elif mt == "DecisionTree":
        f1        = acc - 0.02
        precision = acc
        recall    = acc - 0.03

    elif mt == "SVM":
        f1        = acc - 0.01
        precision = acc + 0.02
        recall    = acc - 0.02

    elif mt == "NaiveBayes":
        # Fixed values per spec.
        f1        = 0.76
        precision = 0.79
        recall    = 0.75

    elif mt == "LogisticRegression":
        f1        = acc - 0.01
        precision = acc + 0.01
        recall    = acc - 0.01

    else:
        f1 = precision = recall = acc

    # Clip derived metrics.
    def _clip(v: float) -> float:
        return round(min(1.0, max(0.0, v)), 4)

    return {
        "accuracy":  _clip(acc),
        "f1":        _clip(f1),
        "precision": _clip(precision),
        "recall":    _clip(recall),
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETER
# ══════════════════════════════════════════════════════════════════════════════

class Interpreter:
    """
    Tree-walking interpreter for the Ayar DSL.

    State triple  σ = (Γ, M, R)  from D1 §4.4:
      self._gamma  — Γ  : global environment (dict).
      self._M      — M  : current experiment results (dict), None outside.
      self._R      — R  : current selection result (str | None).

    Usage
    ─────
        from ayar_lexer  import Lexer
        from ayar_parser import Parser, validate_layout
        tokens = Lexer(source).tokenize()
        validate_layout(tokens)
        ast = Parser(tokens).parse()
        Interpreter(ast, tokens).run()
    """

    def __init__(self, program: ProgramNode, tokens: list = None):
        self._program = program
        self._tokens  = tokens or []

        # ── σ = (Γ, M, R) ────────────────────────────────────────────────────
        self._gamma: Dict[str, Any]             = {}   # Γ
        self._M:     Dict[str, ModelResult]     = {}   # M (active inside exp.)
        self._R:     Optional[str]              = None # R

        # Experiment-scope bookkeeping (not part of the formal state but needed
        # to implement compare, exclude, and report correctly).
        self._compare_metric:    Optional[str]  = None
        self._compare_direction: Optional[str]  = None
        self._ranked_order:      List[str]      = []   # model names after sort
        self._collected_metrics: List[str]      = []   # from collect stmt
        self._analyze_threshold: Optional[float] = None
        self._analyze_underfit_max: Optional[float] = None

        # The most recently completed experiment's results, kept for report.
        self._last_experiment_M: Dict[str, ModelResult] = {}
        self._last_experiment_R: Optional[str]          = None
        self._last_experiment_name: Optional[str]       = None

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        """
        1. Run the TypeChecker.  Abort with a formatted error list on failure.
        2. Execute every top-level declaration in order (E-Sequence rule).
        """
        # ── Step 1: type-check ────────────────────────────────────────────────
        checker = TypeChecker(self._program, self._tokens)
        errors  = checker.check()
        if errors:
            print("\n[Ayar Interpreter] Type checking failed — execution aborted.")
            print(f"  {len(errors)} error(s) found:\n")
            for e in errors:
                print(f"  {e}")
            return

        # ── Step 2: interpret ─────────────────────────────────────────────────
        for decl in self._program.declarations:
            self._exec(decl)

    # ── Statement dispatcher ──────────────────────────────────────────────────

    def _exec(self, node) -> None:
        """Dispatch a node to its handler."""
        if   isinstance(node, DatasetDeclNode):    self._exec_dataset_decl(node)
        elif isinstance(node, SplitStmtNode):      self._exec_split_stmt(node)
        elif isinstance(node, ModelDeclNode):      self._exec_model_decl(node)
        elif isinstance(node, ExperimentDeclNode): self._exec_experiment(node)
        elif isinstance(node, EvalStmtNode):       self._exec_eval_stmt(node)
        elif isinstance(node, ReportStmtNode):     self._exec_report_stmt(node)
        elif isinstance(node, TrainStmtNode):      self._exec_train_stmt(node)
        elif isinstance(node, EvaluateStmtNode):   self._exec_evaluate_stmt(node)
        elif isinstance(node, CollectStmtNode):    self._exec_collect_stmt(node)
        elif isinstance(node, AnalyzeStmtNode):    self._exec_analyze_stmt(node)
        elif isinstance(node, CompareStmtNode):    self._exec_compare_stmt(node)
        elif isinstance(node, ExcludeStmtNode):    self._exec_exclude_stmt(node)
        elif isinstance(node, SelectStmtNode):     self._exec_select_stmt(node)
        else:
            # Unknown / expression nodes — silently skip (already type-checked).
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # TOP-LEVEL DECLARATIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _exec_dataset_decl(self, node: DatasetDeclNode) -> None:
        """
        Rule E-DatasetDecl:
            path ∈ String    id ∉ dom(Γ)
            ─────────────────────────────────────────
            Γ[id ↦ DatasetRecord(path)]
        """
        record = DatasetRecord(name=node.name, path=node.path)
        self._gamma[node.name] = record

    def _exec_split_stmt(self, node: SplitStmtNode) -> None:
        """
        Rule E-SplitStmt:
            id ∈ dom(Γ)    r₁ + r₂ + r₃ = 1.0
            ─────────────────────────────────────────
            Γ[id.train ↦ r₁, id.validation ↦ r₂, id.test ↦ r₃]

        Stores ratios both inside the DatasetRecord.splits dict and as
        dotted keys in Γ for direct lookup.
        """
        dataset_record: DatasetRecord = self._gamma.get(node.dataset)
        if dataset_record is None:
            # Should not happen after type checking, but guard defensively.
            raise AyarRuntimeError(
                f"Dataset '{node.dataset}' referenced in split but not declared."
            )
        for part in node.parts:
            dataset_record.splits[part.role] = part.ratio
            # Also bind dotted key into Γ so E-Train / E-Evaluate can look up
            # iris.train etc. directly.
            self._gamma[f"{node.dataset}.{part.role}"] = part.ratio

    def _exec_model_decl(self, node: ModelDeclNode) -> None:
        """
        Bind the model name to a ModelRecord in Γ.
        Hyperparameter fields are extracted from FieldNode / LiteralNode children.
        """
        hyperparams: Dict[str, Any] = {}
        for fnode in node.fields:
            if isinstance(fnode, FieldNode):
                val_node = fnode.value
                if isinstance(val_node, LiteralNode):
                    hyperparams[fnode.key] = val_node.value
                else:
                    hyperparams[fnode.key] = None   # fallback
        record = ModelRecord(
            model_type  = node.model_type,
            name        = node.name,
            hyperparams = hyperparams,
        )
        self._gamma[node.name] = record

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT BLOCK  (Rule E-ExperimentBlock)
    # ══════════════════════════════════════════════════════════════════════════

    def _exec_experiment(self, node: ExperimentDeclNode) -> None:
        """
        Rule E-ExperimentBlock:
            ⟨s₁;…;sₙ, (Γ, ∅, ⊥)⟩  →  (Γ', M', R')
            Γ[exp_name.best ↦ R']

        The experiment opens with M = ∅ and R = ⊥ (None).
        After all statements execute, R' is exported to Γ as exp_name.best.
        M' is discarded (stack-dynamic lifetime, Sebesta §5.8.2).
        """
        # Save outer state (in case of nested structures, though Ayar has none).
        outer_M                 = self._M
        outer_R                 = self._R
        outer_compare_metric    = self._compare_metric
        outer_compare_direction = self._compare_direction
        outer_ranked_order      = self._ranked_order
        outer_collected         = self._collected_metrics
        outer_threshold         = self._analyze_threshold
        outer_underfit_max      = self._analyze_underfit_max

        # Open fresh experiment scope: M = ∅, R = ⊥.
        self._M                  = {}
        self._R                  = None
        self._compare_metric     = None
        self._compare_direction  = None
        self._ranked_order       = []
        self._collected_metrics  = []
        self._analyze_threshold  = None
        self._analyze_underfit_max = None

        # Execute all statements in sequence (Rule E-Sequence).
        for stmt in node.stmts:
            self._exec(stmt)

        # Export:  Γ[exp_name.best ↦ R']
        best_key = f"{node.name}.best"
        self._gamma[best_key] = self._R   # R' or None

        # Export M snapshot into Γ so that report statements can look up
        # this experiment's results by name, even after later experiments run.
        # Key: exp_name.__metrics__  →  snapshot of M (dict[str, ModelResult])
        metrics_key = f"{node.name}.__metrics__"
        self._gamma[metrics_key] = dict(self._M)

        # Keep results accessible for the top-level report and evaluate stmts.
        self._last_experiment_M    = dict(self._M)
        self._last_experiment_R    = self._R
        self._last_experiment_name = node.name

        # Restore outer state.  M' is intentionally not restored — discarded.
        self._M                  = outer_M
        self._R                  = outer_R
        self._compare_metric     = outer_compare_metric
        self._compare_direction  = outer_compare_direction
        self._ranked_order       = outer_ranked_order
        self._collected_metrics  = outer_collected
        self._analyze_threshold  = outer_threshold
        self._analyze_underfit_max = outer_underfit_max

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT STATEMENTS
    # ══════════════════════════════════════════════════════════════════════════

    def _exec_train_stmt(self, node: TrainStmtNode) -> None:
        """
        Rule E-Train:
            model_name ∈ dom(Γ)
            dataset_ref ∈ dom(Γ)
            dataset_ref.role ≠ test      ← data leakage guard
            simulate(Γ(model_name), Γ(dataset_ref)) = metrics
            ─────────────────────────────────────────────────────
            M[model_name ↦ metrics]

        Stores FULL metrics (all four).  'collect' will restrict them later.
        Also records train_accuracy separately for the overfitting analysis.
        """
        dref = node.dataset_ref
        if isinstance(dref, DatasetRefNode):
            dataset_name = dref.dataset
            role         = dref.role
            # Runtime data-leakage guard (mirrors R5 from the type checker).
            if role == "test":
                raise DataLeakageError(node.model, dataset_name)
        else:
            dataset_name = "unknown"
            role         = "train"

        model_record: ModelRecord = self._gamma.get(node.model)
        if model_record is None:
            raise UndefinedModelError(node.model)

        is_train_split = (role == "train")
        full_metrics   = simulate(model_record, is_train_split)

        # Create or update the entry in M.
        entry = self._M.get(node.model)
        if entry is None:
            entry = ModelResult(
                model_name     = node.model,
                model_type     = model_record.model_type,
                train_accuracy = full_metrics["accuracy"],
                eval_accuracy  = full_metrics["accuracy"],  # updated by evaluate
                metrics        = dict(full_metrics),
            )
            self._M[node.model] = entry
        else:
            # Train called again — overwrite (this is uncommon but legal).
            entry.train_accuracy = full_metrics["accuracy"]
            entry.metrics        = dict(full_metrics)

    def _exec_evaluate_stmt(self, node: EvaluateStmtNode) -> None:
        """
        Rule E-Evaluate:
            model_name ∈ dom(Γ)    model_name ∈ dom(M)
            dataset_ref ∈ dom(Γ)
            eval_metrics(…) = m'
            ─────────────────────────────────────────
            M[model_name ↦ m']

        Refines the metric values stored in M with evaluate-split values
        (no train bonus).  The train_accuracy field is preserved so that
        the overfitting analysis (which compares train vs val accuracy)
        continues to have the training-split value.
        """
        dref = node.dataset_ref
        role = dref.role if isinstance(dref, DatasetRefNode) else "validation"

        model_record: ModelRecord = self._gamma.get(node.model)
        if model_record is None:
            raise UndefinedModelError(node.model)

        # Evaluate on validation/test: always is_train_split=False.
        eval_metrics = simulate(model_record, is_train_split=False)

        entry = self._M.get(node.model)
        if entry is None:
            # Model was evaluated without a prior train; create entry.
            # train_accuracy is set to eval accuracy (no bonus) since there is
            # no training record to contrast with.
            entry = ModelResult(
                model_name     = node.model,
                model_type     = model_record.model_type,
                train_accuracy = eval_metrics["accuracy"],
                eval_accuracy  = eval_metrics["accuracy"],
                metrics        = dict(eval_metrics),
            )
            self._M[node.model] = entry
        else:
            # Update metrics and eval_accuracy; preserve train_accuracy.
            entry.eval_accuracy = eval_metrics["accuracy"]
            entry.metrics       = dict(eval_metrics)

    def _exec_collect_stmt(self, node: CollectStmtNode) -> None:
        """
        Rule E-Collect:
            M' = { n ↦ restrict(M(n), metrics_list)  |  n ∈ dom(M) }

        Restricts every ModelResult.metrics to only the listed metric names.
        This implements the narrowing semantics: after collect, M entries only
        carry the requested metrics.
        """
        self._collected_metrics = list(node.metrics)
        for name, result in self._M.items():
            restricted = {k: v for k, v in result.metrics.items()
                         if k in node.metrics}
            result.metrics = restricted

    def _exec_analyze_stmt(self, node: AnalyzeStmtNode) -> None:
        """
        Rule P-AnalyzeOverfitting:
            for each n ∈ dom(M):
                overfit(n)  ⟺  (M(n)(train_acc) - M(n)(val_acc)) > threshold
                underfit(n) ⟺  M(n)(val_acc) < underfit_max_acc

        Tags each ModelResult in M with .overfit and .underfit booleans.
        No model is removed here — only tagged.
        """
        threshold    = 0.10   # sensible defaults
        underfit_max = 0.70

        for af in node.fields:
            if isinstance(af, AnalyzeFieldNode):
                if af.key == "threshold":
                    threshold    = af.value
                elif af.key == "underfit_max_acc":
                    underfit_max = af.value

        self._analyze_threshold   = threshold
        self._analyze_underfit_max = underfit_max

        for name, result in self._M.items():
            # Use the stored eval_accuracy field so the gap computation is
            # correct even after 'collect' has removed 'accuracy' from metrics.
            val_acc   = result.eval_accuracy
            train_acc = result.train_accuracy
            gap       = train_acc - val_acc
            result.overfit  = gap > threshold
            result.underfit = val_acc < underfit_max

    def _exec_compare_stmt(self, node: CompareStmtNode) -> None:
        """
        Rule P-Compare:
            ranked = sort(dom(M), by: λn. M(n)(metric), order: direction)

        Imposes a total ordering on the models in M.
        higher_is_better → descending sort (best = largest metric).
        lower_is_better  → ascending  sort (best = smallest metric).

        The ranking is stored in self._ranked_order (list of model names
        in ranked order) and as ModelResult.rank_key for each entry.
        """
        self._compare_metric    = node.metric
        self._compare_direction = node.direction

        reverse = (node.direction == "higher_is_better")
        ranked  = sorted(
            self._M.keys(),
            key     = lambda n: self._M[n].metrics.get(node.metric, 0.0),
            reverse = reverse,
        )
        self._ranked_order = ranked
        for pos, name in enumerate(ranked):
            self._M[name].rank_key = pos  # 0 = best rank

    def _exec_exclude_stmt(self, node: ExcludeStmtNode) -> None:
        """
        Rule P-Exclude:
            M' = { n ↦ M(n)  |  ¬overfit(n) ∧ ¬underfit(n) }
            (subject to which conditions are listed)

        Removes models whose tags match any listed condition.
        Also updates self._ranked_order to remove excluded entries.
        """
        conditions = set(node.conditions)
        excluded   = []

        for name in list(self._M.keys()):
            result = self._M[name]
            should_exclude = (
                ("overfit"  in conditions and result.overfit) or
                ("underfit" in conditions and result.underfit)
            )
            if should_exclude:
                excluded.append(name)
                del self._M[name]

        # Maintain ranked order consistency.
        self._ranked_order = [n for n in self._ranked_order
                              if n not in excluded]

    def _exec_select_stmt(self, node: SelectStmtNode) -> None:
        """
        Rule P-SelectBest:
            success: best_name = first(ranked(M'))   →  R = best_name
            failure: dom(M') = ∅                     →  NoEligibleModelError
        """
        if not self._M:
            raise NoEligibleModelError(
                "  All models were excluded.  No model remains in M."
            )

        # Use ranked order if compare was run; otherwise fall back to
        # insertion order of M.
        if self._ranked_order:
            best = self._ranked_order[0]
        else:
            best = next(iter(self._M))

        self._R = best

    # ══════════════════════════════════════════════════════════════════════════
    # TOP-LEVEL EVALUATE (outside experiment)
    # ══════════════════════════════════════════════════════════════════════════

    def _exec_eval_stmt(self, node: EvalStmtNode) -> None:
        """
        'evaluate exp_name.best on dataset.role;'

        Resolves the eval target to a concrete model name via Γ, then runs
        evaluate semantics on that model using the last experiment's M.

        The result is printed to the terminal but does NOT update Γ or any
        persistent state — it is a one-shot observation step.
        """
        # ── Resolve the eval target ───────────────────────────────────────────
        target      = node.target
        model_name: Optional[str] = None

        if isinstance(target, EvalTargetSimpleNode):
            if target.attr is not None:
                # Form: exp_name.best  →  look up in Γ
                key = f"{target.name}.{target.attr}"
                model_name = self._gamma.get(key)
            else:
                # Form: plain identifier — could be a model name or exp.best key.
                model_name = self._gamma.get(target.name, target.name)

        elif isinstance(target, EvalTargetBracketNode):
            key        = f"{target.obj}.{target.attr}"
            model_name = self._gamma.get(key)

        if model_name is None:
            # The eval target resolved to None (best was never set).
            print(
                "[EvalStmt] Warning: eval target resolved to None "
                "(no 'select best' has run yet)."
            )
            return

        # ── Look up the model record in Γ ─────────────────────────────────────
        model_record: ModelRecord = self._gamma.get(model_name)
        if model_record is None or not isinstance(model_record, ModelRecord):
            # Plain-identifier form: the target is already a model name.
            # Try to find it directly.
            for val in self._gamma.values():
                if isinstance(val, ModelRecord) and val.name == model_name:
                    model_record = val
                    break
            if model_record is None:
                print(
                    f"[EvalStmt] Warning: model '{model_name}' not found in Γ."
                )
                return

        # ── Evaluate on the target split ─────────────────────────────────────
        dref = node.dataset_ref
        role = dref.role if isinstance(dref, DatasetRefNode) else "test"

        is_train = (role == "train")
        metrics  = simulate(model_record, is_train_split=is_train)

        # Restrict to collected metrics if available.
        if self._last_experiment_M and model_name in self._last_experiment_M:
            collected = list(
                self._last_experiment_M[model_name].metrics.keys()
            )
            if collected:
                metrics = {k: v for k, v in metrics.items()
                           if k in collected}

        # Print the evaluation result.
        dataset_name = dref.dataset if isinstance(dref, DatasetRefNode) else "?"
        print(f"\n=== FINAL EVALUATION: {model_name} on {dataset_name}.{role} ===")
        print(f"  Model : {model_name} ({model_record.model_type})")
        print(f"  Split : {dataset_name}.{role}")
        print("  Metrics:")
        for metric, value in metrics.items():
            print(f"    {metric:<12}: {value:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════════════

    def _exec_report_stmt(self, node: ReportStmtNode) -> None:
        """
        Produce a formatted terminal report.

        Format (from D1 requirements):
            === REPORT: <name> ===
            Best model : <model_name> (<model_type>)
            Metrics    :
              accuracy : 0.8820
              f1       : 0.8710
            Overfitting analysis:
              <model>  : OK / OVERFIT / UNDERFIT

        Lookup strategy (Fix P1 — multi-experiment correctness):
          The report name is used as an experiment name to look up
          Γ[name.__metrics__] and Γ[name.best].  If those keys exist, this
          report belongs to experiment <name> and uses its own snapshot.
          If they do not exist, fall back to the most recently executed
          experiment's state (self._last_experiment_M / R), which is the
          correct behaviour for single-experiment programs and for reports
          that are named differently from their experiment.
        """
        # ── Parse report fields ───────────────────────────────────────────────
        requested_metrics: List[str] = []
        show_value: Optional[str]    = None

        for rf in node.fields:
            if isinstance(rf, ReportFieldMetricsNode):
                requested_metrics = list(rf.metrics)
            elif isinstance(rf, ReportFieldShowNode):
                show_value = rf.value

        # ── Resolve experiment M and R ────────────────────────────────────────
        # Strategy:
        #  1. Direct match: Γ[report_name.__metrics__] — report named after exp.
        #  2. Prefix scan:  find an experiment whose name is a leading substring
        #     of the report name (e.g. report "iris_report" → exp "iris_exp"
        #     both start with "iris_").  Use the longest matching prefix.
        #  3. Fall back to the most recently completed experiment.
        metrics_key = f"{node.name}.__metrics__"
        best_key    = f"{node.name}.best"

        if metrics_key in self._gamma:
            # Case 1: report name == experiment name.
            experiment_M: Dict[str, Any] = self._gamma[metrics_key]
            best_name: Optional[str]     = self._gamma.get(best_key)
        else:
            # Case 2: scan for an experiment whose name shares the longest
            # common prefix with this report name.
            all_exp_keys = [k for k in self._gamma if k.endswith(".__metrics__")]
            best_exp_key   = None
            best_prefix_len = 0
            report_lower    = node.name.lower()
            for exp_key in all_exp_keys:
                exp_name = exp_key[: -len(".__metrics__")]
                exp_lower = exp_name.lower()
                # Compute length of shared leading characters.
                shared = sum(
                    1 for a, b in zip(report_lower, exp_lower) if a == b
                )
                if shared > best_prefix_len:
                    best_prefix_len = shared
                    best_exp_key    = exp_key
                    best_exp_name   = exp_name

            if best_exp_key is not None and best_prefix_len > 0:
                experiment_M = self._gamma[best_exp_key]
                best_name    = self._gamma.get(f"{best_exp_name}.best")
            else:
                # Case 3: fall back to the most recently completed experiment.
                experiment_M = self._last_experiment_M
                best_name    = self._last_experiment_R

        # ── Header ────────────────────────────────────────────────────────────
        print(f"\n{'=' * (20 + len(node.name))}")
        print(f"=== REPORT: {node.name} ===")
        print(f"{'=' * (20 + len(node.name))}")

        # ── Best model line ───────────────────────────────────────────────────
        if best_name and best_name in experiment_M:
            best_type = experiment_M[best_name].model_type
            print(f"Best model : {best_name} ({best_type})")
        elif best_name:
            print(f"Best model : {best_name}")
        else:
            print("Best model : (none — select best did not complete)")

        # ── Metrics section ───────────────────────────────────────────────────
        if best_name and best_name in experiment_M:
            best_result = experiment_M[best_name]
            # Use requested metrics if specified; otherwise all collected.
            display_metrics = requested_metrics or list(best_result.metrics.keys())
            print("Metrics    :")
            for m in display_metrics:
                val = best_result.metrics.get(m)
                if val is not None:
                    print(f"  {m:<12}: {val:.4f}")
                else:
                    print(f"  {m:<12}: (not collected)")

        # ── Overfitting analysis section ──────────────────────────────────────
        if show_value == "overfitting_analysis" and experiment_M:
            print("Overfitting analysis:")
            for name in experiment_M.keys():
                result = experiment_M[name]
                if result.overfit:
                    tag = "OVERFIT"
                elif result.underfit:
                    tag = "UNDERFIT"
                else:
                    tag = "OK"
                marker = " ← best" if name == best_name else ""
                print(f"  {name:<20}: {tag}{marker}")

        print()   # trailing blank line for readability


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE RUNNER (also used by the demo below)
# ══════════════════════════════════════════════════════════════════════════════

def run_source(source: str) -> None:
    """Lex → validate layout → parse → interpret a source string."""
    from ayar_lexer import Lexer
    from ayar_parser import Parser, validate_layout
    tokens = Lexer(source).tokenize()
    validate_layout(tokens)
    ast = Parser(tokens).parse()
    try:
        Interpreter(ast, tokens).run()
    except AyarRuntimeError as e:
        error_type = type(e).__name__
        # Strip the bracketed prefix the exception embeds in its own message
        # (e.g. "[NoEligibleModelError]\n  ...") so we don't double-print it.
        raw = str(e).strip()
        lines = raw.splitlines()
        if lines and lines[0].startswith("[") and lines[0].endswith("]"):
            content_lines = lines[1:]
        else:
            content_lines = lines
        print(f"\n[Runtime Error \u2014 {error_type}]")
        for line in content_lines:
            print(f"  {line.strip()}" if line.strip() else "")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# DEMO — run when executed directly
# ══════════════════════════════════════════════════════════════════════════════

_DEMO_PROGRAM = """
dataset iris = load("data/iris.csv");

split iris into
    train(0.70),
    validation(0.15),
    test(0.15);

model KNN knn_clf
{
    k = 3,
    distance = "euclidean"
}

model DecisionTree dt_clf
{
    max_depth = 8,
    criterion = "gini"
}

model SVM svm_clf
{
    C = 5,
    kernel = "rbf"
}

model NaiveBayes nb_clf
{
    var_smoothing = 1
}

model LogisticRegression lr_clf
{
    max_iter = 300,
    solver = "lbfgs"
}

experiment iris_exp
{
    train knn_clf on iris.train;
    train dt_clf  on iris.train;
    train svm_clf on iris.train;
    train nb_clf  on iris.train;
    train lr_clf  on iris.train;

    evaluate knn_clf on iris.validation;
    evaluate dt_clf  on iris.validation;
    evaluate svm_clf on iris.validation;
    evaluate nb_clf  on iris.validation;
    evaluate lr_clf  on iris.validation;

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

evaluate iris_exp.best on iris.test;

report iris_report
{
    metrics = [accuracy, precision, recall, f1],
    show    = overfitting_analysis
}
"""

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        filepath = sys.argv[1]
        try:
            source = open(filepath).read()
        except FileNotFoundError:
            print(f"[Error] File not found: {filepath}")
            sys.exit(1)
        print(f"Running Ayar interpreter on '{filepath}'...\n")
        run_source(source)
    else:
        print("Running Ayar interpreter demo...\n")
        run_source(_DEMO_PROGRAM)
