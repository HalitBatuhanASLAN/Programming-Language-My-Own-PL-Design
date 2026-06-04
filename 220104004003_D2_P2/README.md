# Ayar — ML Model Configuration & Training Simulation DSL

**Course:** CSE 341 — Concepts of Programming Languages  
**University:** Gebze Technical University  
**Student:** Halit Batuhan ASLAN — 220104004003  
**Submission:** Part 2 — 22 May 2026  

---

## 1. What is Ayar?

Ayar is a Domain-Specific Language for configuring machine learning models, simulating their training, and selecting the best one based on evaluation metrics. The name comes from an old Turkish word meaning "calibration/tuning", which maps directly to hyperparameter tuning in ML.

The language does **not** run real ML libraries — it simulates the training process and metric generation internally using formula-based deterministic functions per model type.

---

## 2. Project Structure

```
220104004003_D2_P2/
├── README.md                      ← This file
│
├── ── Source Code ──
├── ayar_lexer.py                  ← Lexical analyzer (scanner)
├── ayar_parser.py                 ← Recursive-descent parser + AST node classes
├── ayar_typechecker.py            ← Two-pass type checker (Part 2)
├── ayar_interpreter.py            ← Tree-walking interpreter (Part 2)
│
├── ── Test Runners ──
├── run_tests.py                   ← Part 1 test runner (34 tests, 4 CLI modes)
├── lexerTest.py                   ← Standalone lexer unit tests (10 tests)
├── ayar_test_suite.py             ← Part 2 auto-discovery test suite (28 tests)
│
├── ── D3 Valid Programs ──
├── sample_1.ayar                  ← Iris: KNN + DecisionTree, f1 higher_is_better
├── sample_2.ayar                  ← Spam: SVM + LogisticRegression, recall lower_is_better
├── sample_3.ayar                  ← All 5 models, intentional NoEligibleModelError
├── sample_4.ayar                  ← All 5 models, strict threshold, NoEligibleModelError
├── sample_5.ayar                  ← Type error demo: R2-UnsplitDataset
├── sample_full.ayar               ← D1 spec sample, all 5 model types
├── sample_minimal.ayar            ← Parse-only minimal program (split not declared by design)
├── sample_errors.ayar             ← Multi-error demo (parse errors)
│
├── ── D3 Invalid Programs ──
├── invalid_1.ayar                 ← ParseError: comma between analyze fields
├── invalid_2.ayar                 ← ParseError: missing semicolon after dataset
├── invalid_3.ayar                 ← ParseError: invalid model type (RandomForest)
├── invalid_4.ayar                 ← LayoutError: { on same line as model header
├── invalid_5.ayar                 ← ParseError: unknown top-level keyword (pipeline)
├── invalid_6.ayar                 ← LayoutError: { on same line as report header
│
└── ── Part 2 Stress Test Programs ──
    ├── 2l1_block_comment.ayar         ← Lexer: multi-line /* */ block comment
    ├── 2l2_string_escapes.ayar        ← Lexer: string with special characters
    ├── 2l3_int_zero_field.ayar        ← Lexer: integer literal 0 as field value
    ├── 2l4_all_four_metrics.ayar      ← Lexer: all four metric literals in collect
    ├── 2l5_single_letter_ids.ayar     ← Lexer: single-character identifiers
    ├── 2p1_two_experiments.ayar       ← Parser: two experiment blocks, two reports
    ├── 2p2_report_metrics_only.ayar   ← Parser: report with no show field
    ├── 2p3_split_int_ratios.ayar      ← Parser: integer ratios → ParseError
    ├── 2p4_train_only_select.ayar     ← Parser: minimal experiment (train + select only)
    ├── 2p5_all_field_types.ayar       ← Parser: model with int/float/string/bool fields
    ├── 2tc1_partial_split.ayar        ← TypeChecker: R2-UnsplitDataset (2 errors)
    ├── 2tc2_model_same_name_as_dataset.ayar  ← TypeChecker: R6-NameCollision
    ├── 2tc3_double_collect.ayar       ← TypeChecker: collect called twice in experiment
    ├── 2tc4_compare_uncollected.ayar  ← TypeChecker: R4-UncollectedMetric
    ├── 2tc5_all_five_valid.ayar       ← TypeChecker: valid program, all 5 model types
    ├── 2i1_svm_c_zero.ayar            ← Interpreter: SVM with C=0 (floor behavior)
    ├── 2i2_naive_bayes_fixed_formula.ayar  ← Interpreter: NaiveBayes determinism
    ├── 2i3_exclude_underfit_only.ayar ← Interpreter: exclude if underfit only
    ├── 2i4_lr_iter_950_ceiling.ayar   ← Interpreter: LR ceiling at max_iter=950
    └── 2i5_identical_hyperparams.ayar ← Interpreter: two models, same hyperparams
```

---

## 3. Requirements

- **Python 3.8+** (no external libraries required)
- No installation needed — uses only the Python standard library (`re`, `dataclasses`, `sys`, `os`, `time`)

---

## 4. How to Run

### 4.1 Run a program end-to-end (type check + interpret)

This is the primary entry point for Part 2. The interpreter automatically calls the type checker before execution and aborts if any type errors are found.

```bash
python ayar_interpreter.py sample_1.ayar
python ayar_interpreter.py sample_2.ayar
python ayar_interpreter.py sample_3.ayar
```

Expected outcomes:

| Program | Expected output |
|---------|----------------|
| `sample_1.ayar` | `dt_clf` wins (DecisionTree, accuracy=0.92, f1=0.90), full REPORT |
| `sample_2.ayar` | `svm_clf` wins (SVM, recall=0.74, lower_is_better), full REPORT |
| `sample_3.ayar` | `[Runtime Error — NoEligibleModelError]` (intentional, threshold=0.05) |
| `sample_4.ayar` | `[Runtime Error — NoEligibleModelError]` (intentional, strict threshold) |
| `sample_5.ayar` | `[Ayar Interpreter] Type checking failed` — 2 R2-UnsplitDataset errors |
| `sample_full.ayar` | `dt_deep` wins (DecisionTree), all 5 models in overfitting analysis |

### 4.2 Run with no argument (built-in demo)

```bash
python ayar_interpreter.py
```

Runs the internal `_DEMO_PROGRAM` featuring all 5 model types with `threshold=0.10`. Intended as a quick smoke test.

### 4.3 Parse only — print AST, no type checking or interpretation

```bash
python run_tests.py --file sample_1.ayar
python run_tests.py --file sample_full.ayar
python run_tests.py --file sample_minimal.ayar
```

> **Note:** `sample_minimal.ayar` is a parse-only test file — it parses successfully but intentionally lacks a split statement. Use `run_tests.py --file` to see its AST. Running it through `ayar_interpreter.py` will raise R2-UnsplitDataset errors by design.

### 4.4 View the token stream (lexer output only)

```bash
python run_tests.py --tokens sample_1.ayar
```

### 4.5 Parse a single expression (precedence testing)

```bash
python run_tests.py --expr "3 + 4 * 2"
```

### 4.6 Test invalid programs (parser and layout errors)

Invalid programs must be tested through `run_tests.py`. Running them through `ayar_interpreter.py` raises a raw Python traceback because parse exceptions are not caught at the interpreter entry point.

```bash
python run_tests.py --file invalid_1.ayar   # ParseError: comma between analyze fields
python run_tests.py --file invalid_2.ayar   # ParseError: missing semicolon
python run_tests.py --file invalid_3.ayar   # ParseError: unknown model type RandomForest
python run_tests.py --file invalid_4.ayar   # LayoutError: { on same line as model header
python run_tests.py --file invalid_5.ayar   # ParseError: unknown keyword pipeline
python run_tests.py --file invalid_6.ayar   # LayoutError: { on same line as report header
```

### 4.7 Run the Part 1 full test suite (34 tests)

```bash
python run_tests.py
```

Expected: `34/34 passed`

### 4.8 Run the Part 2 auto-discovery test suite (28 tests)

```bash
python ayar_test_suite.py
```

Expected: `28 passed / 0 failed / 6 skipped`

> The 6 skipped files are `invalid_*.ayar` — these use the `invalid_` prefix which is not auto-classified by the Part 2 suite. They are tested separately by `run_tests.py` (see §4.6 and §4.7).

Optional flags:

```bash
python ayar_test_suite.py --verbose          # show full output for every test
python ayar_test_suite.py --stop-first       # stop on first failure
python ayar_test_suite.py --category lexer   # run only one category
```

Valid `--category` values: `lexer` · `parser` · `typechecker` · `interpreter` · `sample`

### 4.9 Run standalone lexer unit tests

```bash
python lexerTest.py
```

Expected: `10/10 tests passed`

---

## 5. Architecture

The implementation follows a clean four-stage pipeline:

```
Source code (.ayar)
       │
       ▼
 ┌─────────────┐
 │   LEXER     │   ayar_lexer.py
 │  (Scanner)  │   Source text → list of Token objects
 └─────┬───────┘   Token fields: type, value, line_number, column_number
       │
       ▼
 ┌─────────────┐
 │   LAYOUT    │   ayar_parser.py — validate_layout()
 │  VALIDATOR  │   Enforces: { and } must each be on their own line
 └─────┬───────┘   Raises LayoutError with source location if violated
       │
       ▼
 ┌─────────────┐
 │   PARSER    │   ayar_parser.py
 │  (Syntax)   │   Recursive-descent LL(1), one method per EBNF non-terminal
 └─────┬───────┘   Produces an Abstract Syntax Tree (AST)
       │
       ▼
 ┌─────────────┐
 │    TYPE     │   ayar_typechecker.py
 │   CHECKER   │   Two-pass: collect globals → recursive rule check
 └─────┬───────┘   Reports all AyarTypeError violations; aborts execution if any found
       │
       ▼
 ┌─────────────┐
 │ INTERPRETER │   ayar_interpreter.py
 │  (Execute)  │   Tree-walking, state triple (Γ, M, R) per D1 §4.4
 └─────┬───────┘   Simulates training, runs pipeline, produces terminal output
       │
       ▼
   FINAL EVALUATION + REPORT output
```

---

## 6. Component Details

### 6.1 Lexer (`ayar_lexer.py`)

- **Approach:** Single compiled master regex with named groups (maximal munch, Sebesta §4.2)
- **Post-classification:** WORD pattern matches all identifier-shaped strings; post-scan checks frozensets to assign KEYWORD, METRIC_LITERAL, BOOL_LITERAL, or IDENTIFIER (Sebesta §3.3)
- **Pattern ordering:** BLOCK_COMMENT and COMMENT before OP_DIV; FLOAT before INT; two-char operators before single-char
- **Block comment support:** `/* ... */` spanning multiple lines with correct line/column tracking
- **Error handling:** `LexicalError` with exact line and column number

### 6.2 Parser (`ayar_parser.py`)

- **Approach:** Recursive-descent, one `_parse_*` method per EBNF non-terminal
- **Lookahead:** LL(1) — every alternative starts with a distinct keyword token
- **Precedence:** Encoded via call-stack depth (deeper call = tighter binding):
  `parse_expr` → `_parse_and_expr` → `_parse_not_expr` → `_parse_rel_expr` → `_parse_add_expr` → `_parse_mul_expr` → `_parse_unary_expr` → `_parse_primary`
- **Associativity:** Left-associativity via iterative while-loops (not left-recursion, which would cause infinite recursion in top-down parsing)
- **Error handling:** `ParseError` with expected token, actual token, and exact source location
- **Layout rule:** `validate_layout()` pre-pass rejects `{` or `}` on the same line as the preceding token
- **Sebesta references:** §4.4.1 (recursive descent), §4.4 (LL(1) parsing)

### 6.3 Type Checker (`ayar_typechecker.py`)

Two-pass architecture:

| Pass | Method | What it does |
|------|--------|-------------|
| Pass 1 | `_collect_globals()` | Forward scan over all top-level declarations — collects dataset and model names. Hoisting semantics: declaration order does not matter |
| Pass 2 | `_check_node()` | Recursive visitor walk — enforces all six type rules |

**Type rules enforced:**

| Rule tag | Trigger |
|----------|---------|
| `R1-UndeclaredModel` | Model name in `train`/`evaluate` not declared at global level |
| `R2-UnsplitDataset` | Dataset referenced but no corresponding `split` statement |
| `R3-MetricInArithmetic` | `metric` literal inside `+`, `-`, `*`, `/` expression |
| `R4-UncollectedMetric` | `compare by` metric not present in preceding `collect metrics` list |
| `R5-DataLeakage` | `train model on dataset.test` — training on the test split |
| `R6-NameCollision` | Same name declared as both a dataset and a model |

Line numbers are recovered from the token stream via `_LineMap` (AST nodes do not carry line numbers). When no token stream is available, line is reported as `None`.

### 6.4 Interpreter (`ayar_interpreter.py`)

State triple per D1 §4.4:

| Symbol | Python field | Meaning |
|--------|-------------|---------|
| Γ | `self._gamma` | Global environment: dataset records, model records, and exported `exp_name.best` bindings |
| M | `self._M` | Training results dict, created empty at experiment start, discarded at block end (stack-dynamic, Sebesta §5.8.2) |
| R | `self._R` | Selection result: model name chosen by `select best`, or `None` |

**simulate() formulas — Option C (formula-based, Sebesta §1.3.3 reliability):**

| Model type | Accuracy formula | Key hyperparameter |
|-----------|-----------------|-------------------|
| KNN | `max(0.60, 0.94 - k * 0.02)` | `k` |
| DecisionTree | `min(0.60 + max_depth * 0.05, 0.92)` | `max_depth` |
| SVM | `min(0.70 + C * 0.03, 0.94)` | `C` |
| NaiveBayes | `0.78` (fixed) | — |
| LogisticRegression | `min(0.72 + max_iter * 0.0002, 0.91)` | `max_iter` |

Train bonus: `+0.07` applied to accuracy on the training split to simulate overfitting. Evaluate uses base formula values directly. This ensures `analyze overfitting` produces meaningful results (gap = 0.07).

**Runtime errors:**

| Error class | Trigger |
|------------|---------|
| `DataLeakageError` | `train model on dataset.test` |
| `NoEligibleModelError` | All models excluded by `exclude if`; `select best` has no candidates |
| `UndefinedModelError` | Model referenced but not present in Γ |

### 6.5 AST Node Classes

| Node class | Represents |
|-----------|-----------|
| `ProgramNode` | Top-level program (list of declarations) |
| `DatasetDeclNode` | `dataset X = load("file");` |
| `SplitStmtNode` | `split X into train(...), ...;` |
| `ModelDeclNode` | `model KNN name { fields }` |
| `ExperimentDeclNode` | `experiment name { statements }` |
| `TrainStmtNode` | `train model on dataset.role;` |
| `EvaluateStmtNode` | `evaluate model on dataset.role;` |
| `CollectStmtNode` | `collect metrics [...]` |
| `AnalyzeStmtNode` | `analyze overfitting { fields }` |
| `CompareStmtNode` | `compare by metric direction;` |
| `ExcludeStmtNode` | `exclude if condition;` |
| `SelectStmtNode` | `select best;` |
| `EvalStmtNode` | Top-level evaluate statement |
| `ReportStmtNode` | `report name { fields }` |
| `BinOpNode` | Binary operation (e.g. `a + b`) |
| `UnaryOpNode` | Unary operation (e.g. `-x`) |
| `LiteralNode` | Int, float, string, bool, metric literals |
| `IdentNode` | Identifier reference |
| `DotAccessNode` | Dotted access (e.g. `exp.best`) |

---

## 7. Test File Summary

### Valid Programs (D3)

| File | Dataset | Models | Key features tested |
|------|---------|--------|---------------------|
| `sample_1.ayar` | iris | KNN, DecisionTree | All 7 experiment statements, f1 higher_is_better, dotted eval_target, all 4 metrics, report |
| `sample_2.ayar` | spam | SVM, LogisticRegression | recall lower_is_better, collect without accuracy, bracket eval_target |
| `sample_3.ayar` | churn | All 5 | NoEligibleModelError — threshold=0.05 < gap=0.07, all models overfit |
| `sample_4.ayar` | iris | All 5 | NoEligibleModelError — strict threshold |
| `sample_5.ayar` | — | KNN | R2-UnsplitDataset type error demo |
| `sample_full.ayar` | iris | All 5 | D1 spec sample, dotted eval_target, all 5 model types |
| `sample_minimal.ayar` | wine | KNN | Parse-only: AST structure demo |

### Invalid Programs (D3)

| File | Error type | Deliberate mistake |
|------|-----------|-------------------|
| `invalid_1.ayar` | ParseError | Comma between analyze fields |
| `invalid_2.ayar` | ParseError | Missing `;` after dataset declaration |
| `invalid_3.ayar` | ParseError | Unknown model type `RandomForest` |
| `invalid_4.ayar` | LayoutError | `{` on same line as model header |
| `invalid_5.ayar` | ParseError | Unknown top-level keyword `pipeline` |
| `invalid_6.ayar` | LayoutError | `{` on same line as report header |

### Part 2 Stress Tests (auto-discovered by `ayar_test_suite.py`)

| Prefix | Category | Count | Expected outcome |
|--------|----------|-------|-----------------|
| `2l*` | Lexer | 5 | Clean run |
| `2p*` (except `2p3`) | Parser | 4 | Clean run |
| `2p3*` | Parser | 1 | ParseError (integer split ratios) |
| `2tc1`–`2tc4` | TypeChecker | 4 | At least 1 AyarTypeError |
| `2tc5` | TypeChecker | 1 | Clean run (valid baseline) |
| `2i*` | Interpreter | 5 | Clean run with output |

---

## 8. Operator Precedence (low → high)

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `or` | Left |
| 2 | `and` | Left |
| 3 | `not` | Right-recursive (unary) |
| 4 | `== != < <= > >=` | Non-chaining |
| 5 | `+ -` | Left |
| 6 | `* /` | Left |
| 7 | Unary `-` | Right-recursive (unary) |

---

## 9. Language Features Status

| Feature | Part 1 | Part 2 |
|---------|--------|--------|
| Dataset declaration and split | ✅ | ✅ |
| 5 model types with hyperparameters | ✅ | ✅ |
| Experiment block — all 7 statement types | ✅ | ✅ |
| 3 eval_target forms (plain, dotted, bracket) | ✅ | ✅ |
| Report statement | ✅ | ✅ |
| Expression language (precedence, associativity) | ✅ | ✅ |
| Layout rule (braces on own line) | ✅ | ✅ |
| Informative error messages with line/column | ✅ | ✅ |
| AST dump (`run_tests.py --file`) | ✅ | ✅ |
| Block comment support (`/* ... */`) | ❌ | ✅ |
| Type checker (R1–R6 rules) | ❌ | ✅ |
| Formula-based training simulation | ❌ | ✅ |
| Overfitting analysis and model selection pipeline | ❌ | ✅ |
| Data leakage detection at runtime | ❌ | ✅ |
| NoEligibleModelError (loud failure design) | ❌ | ✅ |
| Named experiment report lookup | ❌ | ✅ |

---

## 10. Known Limitations

- **Expression sub-language** — the full `<expr>` hierarchy is parsed and AST-built correctly but has no semantic evaluation in the interpreter. Arithmetic expressions are not used in any D3 programs. The type checker's R3 rule (metric literals in arithmetic) can only be triggered via a synthetic AST, not through the parser, because the grammar does not place expressions in executable field positions.
- **`sample_minimal.ayar`** — parses successfully but lacks a `split` statement by design. Use `run_tests.py --file sample_minimal.ayar` to inspect the AST. Running it through `ayar_interpreter.py` raises R2-UnsplitDataset type errors intentionally.
- **`invalid_*.ayar` files** — must be tested through `run_tests.py --file`, not `ayar_interpreter.py`. The interpreter does not catch parse exceptions at the entry point.