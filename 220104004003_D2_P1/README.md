# Ayar — ML Model Configuration & Training Simulation DSL

**Course:** CSE 341 — Concepts of Programming Languages  
**University:** Gebze Technical University  
**Student:** Halit Batuhan ASLAN — 220104004003  
**Submission:** Part 1 — 8 May 2026  

---

## 1. What is Ayar?

Ayar is a Domain-Specific Language for configuring machine learning models, simulating their training, and selecting the best one based on evaluation metrics. The name comes from Turkish meaning "calibration/tuning", which maps directly to hyperparameter tuning in ML.

The language does **not** run real ML libraries — it simulates the training process and metric generation internally.

## 2. Project Structure

```
ayar/
├── README.md                ← This file
│
├── ── Source Code ──
├── ayar_lexer.py            ← Lexical analyzer (scanner)
├── ayar_parser.py           ← Recursive-descent parser + AST node classes
│
├── ── Test Runners ──
├── run_tests.py             ← Main test runner (34 automated tests, 4 CLI modes)
├── lexerTest.py             ← Standalone lexer unit tests (10 tests)
│
├── ── Valid Programs (D3) ──
├── sample_1.ayar            ← Iris classification — KNN + DecisionTree, dotted eval_target
├── sample_2.ayar            ← Spam detection — SVM + NaiveBayes + LogisticRegression, bracket eval_target
├── sample_3.ayar            ← Churn prediction — all 5 model types, plain eval_target
│
├── ── Additional Valid Programs ──
├── sample_4.ayar            ← Full grammar coverage — exercises every grammar rule at least once
├── sample_5.ayar            ← Minimal program — smallest valid Ayar program (smoke test)
├── sample_full.ayar         ← D1 spec sample — exact program from the Design Specification, all 5 model types
├── sample_minimal.ayar      ← Extended minimal — adds analyze block and dotted eval_target
│
├── ── Invalid Programs (D3) ──
├── invalid_1.ayar           ← Error: comma between analyze fields (old style, now illegal)
├── invalid_2.ayar           ← Error: missing semicolon after dataset declaration
├── invalid_3.ayar           ← Error: invalid model type (RandomForest)
├── invalid_4.ayar           ← Error: layout violation — { on same line as model header
├── invalid_5.ayar           ← Error: unknown top-level keyword (pipeline)
├── invalid_6.ayar           ← Error: layout violation — { on same line as report header
│
├── ── Error Demonstration ──
└── sample_errors.ayar       ← Multi-error file — uncomment one block at a time to trigger different errors
```

## 3. Requirements

- **Python 3.8+** (no external libraries required)
- No installation needed — the lexer and parser use only the Python standard library (`re`, `dataclasses`)

## 4. How to Run

### Run the full automated test suite (34 tests):

```bash
python3 run_tests.py
```

Expected output: `34/34 tests passed`.

### Parse a single `.ayar` file and print its AST:

```bash
python3 run_tests.py --file sample_1.ayar
python3 run_tests.py --file sample_full.ayar
python3 run_tests.py --file sample_minimal.ayar
```

### View the token stream (lexer output only):

```bash
python3 run_tests.py --tokens sample_1.ayar
```

### Parse a single expression (for testing precedence):

```bash
python3 run_tests.py --expr "3 + 4 * 2"
```

### Run standalone lexer tests (10 tests):

```bash
python3 lexerTest.py
```

### Test an invalid program (should print error with line/column):

```bash
python3 run_tests.py --file invalid_1.ayar
python3 run_tests.py --file invalid_3.ayar
python3 run_tests.py --file invalid_4.ayar
```

## 5. Architecture

The implementation follows a clean three-stage pipeline:

```
Source code (.ayar)
       │
       ▼
 ┌─────────────┐
 │   LEXER     │   ayar_lexer.py
 │  (Scanner)  │   Converts source text → list of Token objects
 └─────┬───────┘   Each token has: type, value, line_number, column_number
       │
       ▼
 ┌─────────────┐
 │  VALIDATOR   │   ayar_parser.py (validate_layout function)
 │  (Layout)   │   Enforces: { and } must each be on their own line
 └─────┬───────┘   Raises LayoutError if violated
       │
       ▼
 ┌─────────────┐
 │   PARSER    │   ayar_parser.py
 │  (Syntax)   │   Recursive-descent, LL(1), one method per EBNF non-terminal
 └─────┬───────┘   Produces an Abstract Syntax Tree (AST)
       │
       ▼
   AST output
   (printable via dump_ast)
```

### 5.1 Lexer (`ayar_lexer.py`)

- **Approach:** Single compiled master regex with named groups (maximal munch)
- **Token classification:** WORD pattern matches all identifier-shaped strings, then a post-classification step checks frozen sets to assign KEYWORD, METRIC_LITERAL, BOOL_LITERAL, or IDENTIFIER
- **Pattern ordering:** COMMENT before OP_DIV, FLOAT before INT, two-char operators (==, !=, <=, >=) before single-char ones
- **Error handling:** `LexicalError` with exact line and column number
- **Sebesta reference:** §4.2 (maximal munch), §3.3 (reserved word recognition)

### 5.2 Parser (`ayar_parser.py`)

- **Approach:** Recursive-descent, one `_parse_*` method per EBNF non-terminal
- **Lookahead:** LL(1) — every alternative starts with a distinct keyword
- **Precedence:** Encoded via call hierarchy depth (deeper = tighter binding):
  `parse_expr` → `_parse_and_expr` → `_parse_not_expr` → `_parse_rel_expr` → `_parse_add_expr` → `_parse_mul_expr` → `_parse_unary_expr` → `_parse_primary`
- **Associativity:** Left-associativity via iterative while-loops (not left-recursion, which would cause infinite recursion in top-down parsing)
- **Error handling:** `ParseError` with expected token, actual token, and exact line/column
- **Layout rule:** `validate_layout()` runs before parsing — rejects `{` or `}` on the same line as the preceding token
- **Sebesta reference:** §4.4.1 (recursive descent), §4.4 (LL(1) parsing)

### 5.3 AST Node Classes

Every construct has its own Python class:

| Node Class | Represents |
|---|---|
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
| `BinOpNode` | Binary operation (e.g., `a + b`) |
| `UnaryOpNode` | Unary operation (e.g., `-x`) |
| `LiteralNode` | Int, float, string, bool, metric literals |
| `IdentNode` | Identifier reference |
| `DotAccessNode` | Dotted access (e.g., `exp.best`) |

## 6. Test File Summary

### Valid Programs

| File | Dataset | Models | Key Features Tested |
|---|---|---|---|
| `sample_1.ayar` | iris | KNN, DecisionTree | int/string/bool fields, all 7 experiment statements, dotted eval_target, report |
| `sample_2.ayar` | spam | SVM, NaiveBayes, LogisticRegression | float fields, `lower_is_better`, single-condition exclude, bracket eval_target, string show value |
| `sample_3.ayar` | churn | All 5 model types | Maximum coverage, both analyze fields, plain eval_target |
| `sample_4.ayar` | iris | All 5 model types | Full grammar rule coverage, bracket eval_target |
| `sample_5.ayar` | wine | KNN | Smallest valid program — smoke test (3 AST nodes) |
| `sample_full.ayar` | iris | All 5 model types | D1 spec sample program, dotted eval_target |
| `sample_minimal.ayar` | wine | KNN | Extended minimal with analyze block and dotted eval_target |

### Invalid Programs

| File | Error Type | Deliberate Mistake |
|---|---|---|
| `invalid_1.ayar` | ParseError | Comma between analyze fields (old style, now illegal) |
| `invalid_2.ayar` | ParseError | Missing `;` after dataset declaration |
| `invalid_3.ayar` | ParseError | Unknown model type `RandomForest` |
| `invalid_4.ayar` | LayoutError | `{` on same line as model header |
| `invalid_5.ayar` | ParseError | Unknown top-level keyword `pipeline` |
| `invalid_6.ayar` | LayoutError | `{` on same line as report header |
| `sample_errors.ayar` | Mixed | Multi-error demo file — uncomment one block at a time |

## 7. Language Features (Part 1 Scope)

| Feature | Status |
|---|---|
| Dataset declaration and loading | ✅ Implemented |
| Split statement (train/validation/test) | ✅ Implemented |
| 5 model types (KNN, DecisionTree, SVM, NaiveBayes, LogisticRegression) | ✅ Implemented |
| Experiment block with all 7 statement types | ✅ Implemented |
| 3 eval_target forms (plain, dotted, bracket) | ✅ Implemented |
| Report statement | ✅ Implemented |
| Expression language with precedence & associativity | ✅ Implemented |
| Layout rule (braces on own line) | ✅ Implemented |
| Informative error messages with line/column | ✅ Implemented |
| AST dump (`run_tests.py --file`) | ✅ Implemented |

## 8. Operator Precedence (low → high)

| Level | Operators | Associativity |
|---|---|---|
| 1 | `or` | Left |
| 2 | `and` | Left |
| 3 | `not` | Right (unary) |
| 4 | `==  !=  <  <=  >  >=` | Non-associative |
| 5 | `+  -` | Left |
| 6 | `*  /` | Left |
| 7 | Unary `-` | Right (unary) |
| 8 | Primary (literals, identifiers, parenthesized) | — |

## 9. Example

```
dataset iris = load("iris.csv");
split iris into train(0.70), validation(0.15), test(0.15);

model KNN knn_small
{
    k = 3,
    distance = "euclidean"
}

experiment iris_exp
{
    train knn_small on iris.train;
    evaluate knn_small on iris.validation;
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

report summary
{
    metrics = [accuracy, f1],
    show = overfitting_analysis
}
```

## 10. Known Limitations (Part 1)

- **No type checker yet** — will be added in Part 2 (strong typing, name equivalence for metric type, int→float coercion)
- **No interpreter yet** — will be added in Part 2 (static scoping, simulated training, metric generation)
- **No runtime error detection yet** — data leakage (train on test), division by zero, undefined model references will be caught in Part 2