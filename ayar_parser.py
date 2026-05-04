"""
ayar_parser.py
==============
Recursive-Descent Parser for the 'ayar' DSL
(CSE341 – Concepts of Programming Languages)

Theory reference: Sebesta, "Concepts of Programming Languages" 12th ed., Chapter 4
  – Section 4.4  : Recursive-Descent Parsing
  – Section 4.5  : Bottom-up / operator-precedence (we handle this top-down here)
  – Section 4.1  : Introduction to Syntax Description (EBNF)

Token types this parser expects  (must match ayar_lexer.py exactly)
─────────────────────────────────────────────────────────────────────
  KEYWORD        – every reserved word, model type, split role, compare dir…
  IDENTIFIER     – user-defined names
  INT_LITERAL    – e.g.  5
  FLOAT_LITERAL  – e.g.  0.15
  STRING_LITERAL – e.g.  "iris.csv"  (value INCLUDES the surrounding quotes)
  BOOL_LITERAL   – true | false
  METRIC_LITERAL – accuracy | precision | recall | f1
  OP_ASSIGN      – =
  OP_EQ          – ==
  OP_NEQ         – !=
  OP_LT          – <
  OP_LTE         – <=
  OP_GT          – >
  OP_GTE         – >=
  OP_PLUS        – +
  OP_MINUS       – -
  OP_MUL         – *
  OP_DIV         – /
  LBRACE         – {
  RBRACE         – }
  LPAREN         – (
  RPAREN         – )
  LBRACKET       – [
  RBRACKET       – ]
  COMMA          – ,
  SEMICOLON      – ;
  DOT            – .
  EOF            – sentinel appended by the parser itself

Design overview
───────────────
  1. Every EBNF non-terminal has exactly ONE corresponding parse_*() method.
  2. The token stream is consumed left-to-right with one-token look-ahead
     (self.current).
  3. Operator precedence is encoded in the CALL HIERARCHY:
         parse_expr            (lowest  – "or")
           └─ _parse_and_expr             ("and")
               └─ _parse_not_expr         ("not")
                   └─ _parse_rel_expr     (== != < <= > >=)
                       └─ _parse_add_expr (+ -)
                           └─ _parse_mul_expr (* /)
                               └─ _parse_unary_expr (unary -)
                                   └─ _parse_primary (literals, id, parens)
  4. Left-associativity is achieved by the iterative while-loop pattern
     inside _parse_add_expr and _parse_mul_expr (Sebesta §4.4).
  5. ParseError carries exact line + column from the Token's metadata.
"""

from ayar_lexer import Token   # reuse the dataclass from the student's lexer

# ═══════════════════════════════════════════════════════════════════════════════
# SENTINEL EOF TOKEN
# ═══════════════════════════════════════════════════════════════════════════════

_EOF = Token(type="EOF", value="", line_number=0, column_number=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PARSE ERROR
# ═══════════════════════════════════════════════════════════════════════════════

class ParseError(Exception):
    """
    Raised when the parser encounters an unexpected token.

    The message contains:
      • what was EXPECTED  (human-readable description)
      • what was RECEIVED  (type + value of the actual Token)
      • the exact line and column from the Token's metadata

    Satisfies Sebesta's discussion of error detection (§4.1.3).
    """
    def __init__(self, expected: str, actual_token: Token, extra: str = ""):
        loc = f"line {actual_token.line_number}, col {actual_token.column_number}"
        msg = (
            f"\n[ParseError] at {loc}\n"
            f"  Expected : {expected}\n"
            f"  Received : token type={actual_token.type!r}, "
            f"value={actual_token.value!r}"
        )
        if extra:
            msg += f"\n  Note     : {extra}"
        super().__init__(msg)
        self.expected     = expected
        self.actual_token = actual_token


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT ERROR  +  PRE-PARSE LAYOUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class LayoutError(Exception):
    """
    Raised when a '{' or '}' appears on the same source line as the token
    that immediately precedes it.

    The 'ayar' grammar requires every curly brace to occupy its own line:

        LEGAL                       ILLEGAL
        ─────────────────────────   ─────────────────────────
        model KNN k                 model KNN k {
        {                               n = 5
            n = 5                   }
        }

    This is a LAYOUT rule (sometimes called an "off-side" constraint).
    It cannot be expressed in the EBNF itself, so it is enforced here as a
    dedicated pre-parse validation pass rather than inside the parser proper.
    Keeping it separate means:
      • The parser stays clean — it never needs to check line numbers.
      • The error message can pinpoint exactly which brace is wrong.
    """
    def __init__(self, brace: str, token, prev_token):
        loc = f"line {token.line_number}, col {token.column_number}"
        msg = (
            f"\n[LayoutError] at {loc}\n"
            f"  Rule     : '{{' and '}}' must each be on their own line\n"
            f"  Found    : '{brace}' on the same line as the preceding token\n"
            f"  Preceding: token type={prev_token.type!r}, "
            f"value={prev_token.value!r} "
            f"at line {prev_token.line_number}, col {prev_token.column_number}"
        )
        super().__init__(msg)
        self.token      = token
        self.prev_token = prev_token


def validate_layout(tokens: list) -> None:
    """
    Walk the token list once and raise LayoutError for the first '{' or '}'
    that shares a line with its immediately preceding (non-EOF) token.

    Rule applied to BOTH braces:
        prev_token.line_number == brace_token.line_number  →  LayoutError

    Call this BEFORE constructing a Parser, so layout problems are caught
    and reported before any grammar analysis begins.

    Parameters
    ----------
    tokens : the list returned by Lexer.tokenize()  (may include EOF at end)
    """
    # Filter out any whitespace/comment/newline tokens that the lexer might
    # have left in; we only want real tokens when comparing line numbers.
    real = [t for t in tokens if t.type not in ("NEWLINE", "WHITESPACE", "COMMENT")]

    for i, tok in enumerate(real):
        if tok.type not in ("LBRACE", "RBRACE"):
            continue
        if i == 0:
            # A brace as the very first token has no predecessor — allow it.
            continue
        prev = real[i - 1]
        if prev.type == "EOF":
            continue
        if prev.line_number == tok.line_number:
            raise LayoutError(tok.value, tok, prev)


# ═══════════════════════════════════════════════════════════════════════════════
# AST NODE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class ASTNode:
    """Abstract base – all concrete nodes inherit from this."""
    def dump(self, indent: int = 0) -> str:
        raise NotImplementedError


def _ind(n: int) -> str:
    return "  " * n


# ── program ────────────────────────────────────────────────────────────────────

class ProgramNode(ASTNode):
    def __init__(self, declarations: list):
        self.declarations = declarations
    def dump(self, indent=0):
        lines = [_ind(indent) + "Program"]
        for d in self.declarations:
            lines.append(d.dump(indent + 1))
        return "\n".join(lines)


# ── dataset ────────────────────────────────────────────────────────────────────

class DatasetDeclNode(ASTNode):
    """dataset <name> = load(<path>);"""
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path          # stored WITHOUT surrounding quotes
    def dump(self, indent=0):
        return _ind(indent) + f"DatasetDecl(name={self.name!r}, path={self.path!r})"


class SplitStmtNode(ASTNode):
    def __init__(self, dataset: str, parts: list):
        self.dataset = dataset
        self.parts   = parts
    def dump(self, indent=0):
        lines = [_ind(indent) + f"SplitStmt(dataset={self.dataset!r})"]
        for p in self.parts:
            lines.append(p.dump(indent + 1))
        return "\n".join(lines)


class SplitPartNode(ASTNode):
    def __init__(self, role: str, ratio: float):
        self.role  = role
        self.ratio = ratio
    def dump(self, indent=0):
        return _ind(indent) + f"SplitPart(role={self.role!r}, ratio={self.ratio})"


# ── model ──────────────────────────────────────────────────────────────────────

class ModelDeclNode(ASTNode):
    def __init__(self, model_type: str, name: str, fields: list):
        self.model_type = model_type
        self.name       = name
        self.fields     = fields
    def dump(self, indent=0):
        lines = [_ind(indent) +
                 f"ModelDecl(type={self.model_type!r}, name={self.name!r})"]
        for f in self.fields:
            lines.append(f.dump(indent + 1))
        return "\n".join(lines)


class FieldNode(ASTNode):
    def __init__(self, key: str, value: ASTNode):
        self.key   = key
        self.value = value
    def dump(self, indent=0):
        return (_ind(indent) + f"Field(key={self.key!r})\n"
                + self.value.dump(indent + 1))


# ── experiment ─────────────────────────────────────────────────────────────────

class ExperimentDeclNode(ASTNode):
    def __init__(self, name: str, stmts: list):
        self.name  = name
        self.stmts = stmts
    def dump(self, indent=0):
        lines = [_ind(indent) + f"ExperimentDecl(name={self.name!r})"]
        for s in self.stmts:
            lines.append(s.dump(indent + 1))
        return "\n".join(lines)


class TrainStmtNode(ASTNode):
    def __init__(self, model: str, dataset_ref):
        self.model       = model
        self.dataset_ref = dataset_ref
    def dump(self, indent=0):
        return (_ind(indent) + f"TrainStmt(model={self.model!r})\n"
                + self.dataset_ref.dump(indent + 1))


class EvaluateStmtNode(ASTNode):
    def __init__(self, model: str, dataset_ref):
        self.model       = model
        self.dataset_ref = dataset_ref
    def dump(self, indent=0):
        return (_ind(indent) + f"EvaluateStmt(model={self.model!r})\n"
                + self.dataset_ref.dump(indent + 1))


class DatasetRefNode(ASTNode):
    def __init__(self, dataset: str, role: str):
        self.dataset = dataset
        self.role    = role
    def dump(self, indent=0):
        return _ind(indent) + f"DatasetRef({self.dataset!r}.{self.role!r})"


class CollectStmtNode(ASTNode):
    def __init__(self, metrics: list):
        self.metrics = metrics
    def dump(self, indent=0):
        return _ind(indent) + f"CollectStmt(metrics={self.metrics})"


class AnalyzeStmtNode(ASTNode):
    def __init__(self, fields: list):
        self.fields = fields
    def dump(self, indent=0):
        lines = [_ind(indent) + "AnalyzeStmt"]
        for f in self.fields:
            lines.append(f.dump(indent + 1))
        return "\n".join(lines)


class AnalyzeFieldNode(ASTNode):
    def __init__(self, key: str, value: float):
        self.key   = key
        self.value = value
    def dump(self, indent=0):
        return _ind(indent) + f"AnalyzeField({self.key!r} = {self.value})"


class CompareStmtNode(ASTNode):
    def __init__(self, metric: str, direction: str):
        self.metric    = metric
        self.direction = direction
    def dump(self, indent=0):
        return (_ind(indent) +
                f"CompareStmt(metric={self.metric!r}, "
                f"direction={self.direction!r})")


class ExcludeStmtNode(ASTNode):
    def __init__(self, conditions: list):
        self.conditions = conditions
    def dump(self, indent=0):
        return _ind(indent) + f"ExcludeStmt(conditions={self.conditions})"


class SelectStmtNode(ASTNode):
    def dump(self, indent=0):
        return _ind(indent) + "SelectStmt(best)"


# ── top-level evaluate ─────────────────────────────────────────────────────────

class EvalStmtNode(ASTNode):
    def __init__(self, target, dataset_ref):
        self.target      = target
        self.dataset_ref = dataset_ref
    def dump(self, indent=0):
        lines = [_ind(indent) + "EvalStmt"]
        lines.append(self.target.dump(indent + 1))
        lines.append(self.dataset_ref.dump(indent + 1))
        return "\n".join(lines)


class EvalTargetSimpleNode(ASTNode):
    """
    Covers both:
      <identifier>                 – e.g.  myModel
      <identifier> "." <identifier> – e.g.  comparison.best   (D1 sample)

    D1 EBNF lists only <identifier> in the simple branch, but the D1 sample
    program uses  "evaluate comparison.best on iris.test;"  without brackets,
    so the simple branch must also accept a dotted access.
    attr is None when there is no dot.
    """
    def __init__(self, name: str, attr: str = None):
        self.name = name
        self.attr = attr   # None  → plain identifier,  str → dotted access
    def dump(self, indent=0):
        if self.attr is not None:
            return _ind(indent) + f"EvalTarget({self.name!r}.{self.attr!r})"
        return _ind(indent) + f"EvalTarget(id={self.name!r})"


class EvalTargetBracketNode(ASTNode):
    def __init__(self, obj: str, attr: str):
        self.obj  = obj
        self.attr = attr
    def dump(self, indent=0):
        return _ind(indent) + f"EvalTarget([{self.obj!r}.{self.attr!r}])"


# ── report ─────────────────────────────────────────────────────────────────────

class ReportStmtNode(ASTNode):
    def __init__(self, name: str, fields: list):
        self.name   = name
        self.fields = fields
    def dump(self, indent=0):
        lines = [_ind(indent) + f"ReportStmt(name={self.name!r})"]
        for f in self.fields:
            lines.append(f.dump(indent + 1))
        return "\n".join(lines)


class ReportFieldMetricsNode(ASTNode):
    def __init__(self, metrics: list):
        self.metrics = metrics
    def dump(self, indent=0):
        return _ind(indent) + f"ReportField(metrics={self.metrics})"


class ReportFieldShowNode(ASTNode):
    def __init__(self, value: str):
        self.value = value
    def dump(self, indent=0):
        return _ind(indent) + f"ReportField(show={self.value!r})"


# ── expressions ────────────────────────────────────────────────────────────────

class BinOpNode(ASTNode):
    """
    Binary operation.  Left-associativity is baked in by construction:
    the iterative loop in _parse_add_expr / _parse_mul_expr always makes
    the already-accumulated left subtree the LEFT child of the new BinOpNode.

    Example for  a + b + c:
        step 1: node = a
        step 2: node = BinOpNode('+', a, b)
        step 3: node = BinOpNode('+', BinOpNode('+',a,b), c)
    """
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op    = op
        self.left  = left
        self.right = right
    def dump(self, indent=0):
        return "\n".join([
            _ind(indent) + f"BinOp({self.op!r})",
            self.left.dump(indent + 1),
            self.right.dump(indent + 1),
        ])


class UnaryOpNode(ASTNode):
    def __init__(self, op: str, operand: ASTNode):
        self.op      = op
        self.operand = operand
    def dump(self, indent=0):
        return "\n".join([
            _ind(indent) + f"UnaryOp({self.op!r})",
            self.operand.dump(indent + 1),
        ])


class LiteralNode(ASTNode):
    """
    kind  : 'int' | 'float' | 'bool' | 'string' | 'metric'
    value : Python-typed  (int, float, bool, str)
    """
    def __init__(self, kind: str, value):
        self.kind  = kind
        self.value = value
    def dump(self, indent=0):
        return _ind(indent) + f"Literal(kind={self.kind!r}, value={self.value!r})"


class IdentifierNode(ASTNode):
    def __init__(self, name: str):
        self.name = name
    def dump(self, indent=0):
        return _ind(indent) + f"Identifier({self.name!r})"


class AttributeAccessNode(ASTNode):
    def __init__(self, obj: str, attr: str):
        self.obj  = obj
        self.attr = attr
    def dump(self, indent=0):
        return _ind(indent) + f"AttributeAccess({self.obj!r}.{self.attr!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (sets used for fast membership tests inside parse methods)
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_TYPES  = {"KNN", "DecisionTree", "SVM", "NaiveBayes", "LogisticRegression"}
_SPLIT_ROLES  = {"train", "validation", "test"}
_COMPARE_DIRS = {"higher_is_better", "lower_is_better"}
_REL_OP_TYPES = {"OP_EQ", "OP_NEQ", "OP_LT", "OP_LTE", "OP_GT", "OP_GTE"}


# ═══════════════════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    """
    Recursive-Descent Parser for the 'ayar' DSL.

    Usage
    ─────
        from ayar_lexer import Lexer
        tokens = Lexer(source_code).tokenize()
        ast    = Parser(tokens).parse()
        Parser.print_tree(ast)

    Primitives
    ──────────
    _peek()    – inspect current token WITHOUT advancing (one-token look-ahead)
    _advance() – consume and return the current token, then move forward
    _consume() – like _advance() but ASSERTS the token matches first;
                 raises ParseError if not
    """

    def __init__(self, tokens: list):
        # Drop any residual NEWLINE/WHITESPACE/COMMENT tokens just in case.
        self.tokens  = [t for t in tokens
                        if t.type not in ("NEWLINE", "WHITESPACE", "COMMENT")]
        self.pos     = 0
        self.current = self.tokens[0] if self.tokens else _EOF

    # ── primitives ─────────────────────────────────────────────────────────────

    def _peek(self) -> Token:
        return self.current

    def _peek_next(self) -> Token:
        nxt = self.pos + 1
        return self.tokens[nxt] if nxt < len(self.tokens) else _EOF

    def _advance(self) -> Token:
        tok      = self.current
        self.pos += 1
        self.current = (self.tokens[self.pos]
                        if self.pos < len(self.tokens) else _EOF)
        return tok

    def _consume(self, expected_type: str, expected_value: str = None) -> Token:
        """
        Assert the current token matches (type[, value]), then advance.

        This is the primary error-detection gate (Sebesta §4.1.3).
        A failed assertion raises ParseError with the exact source location.
        """
        tok      = self._peek()
        type_ok  = (tok.type == expected_type)
        value_ok = (expected_value is None) or (tok.value == expected_value)
        if not (type_ok and value_ok):
            desc = (f"token type={expected_type!r}"
                    + (f" with value={expected_value!r}" if expected_value else ""))
            raise ParseError(desc, tok)
        return self._advance()

    def _is(self, type_: str, value: str = None) -> bool:
        tok = self._peek()
        return tok.type == type_ and (value is None or tok.value == value)

    def _consume_keyword(self, kw: str) -> Token:
        """Consume a KEYWORD token with a specific value."""
        tok = self._peek()
        if tok.type != "KEYWORD" or tok.value != kw:
            raise ParseError(f"keyword {kw!r}", tok)
        return self._advance()

    # ── string helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _strip_quotes(s: str) -> str:
        """
        The student's lexer keeps the surrounding double-quotes in the
        STRING_LITERAL token value (e.g. '"iris.csv"').
        Strip them so the AST holds the raw content.
        """
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    # ══════════════════════════════════════════════════════════════════════════
    # TOP-LEVEL ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def parse(self) -> ProgramNode:
        """
        <program> ::= { <top_level_decl> }

        Step 1 - layout validation: every '{' and '}' must be on its own line.
                 validate_layout() raises LayoutError immediately if not.
        Step 2 - recursive-descent grammar parse.

        Outer { } in EBNF = zero-or-more.  We loop until the EOF sentinel.
        """
        validate_layout(self.tokens)          # layout check before grammar
        declarations = []
        while not self._is("EOF"):
            declarations.append(self._parse_top_level_decl())
        return ProgramNode(declarations)

    # ── top-level dispatch ─────────────────────────────────────────────────────

    def _parse_top_level_decl(self) -> ASTNode:
        """
        <top_level_decl> ::= <dataset_decl> | <split_stmt>  | <model_decl>
                           | <experiment_decl> | <eval_stmt> | <report_stmt>

        Predictive (LL-1) dispatch on the value of the current KEYWORD token.
        """
        tok = self._peek()
        kw  = tok.value

        if   kw == "dataset":    return self._parse_dataset_decl()
        elif kw == "split":      return self._parse_split_stmt()
        elif kw == "model":      return self._parse_model_decl()
        elif kw == "experiment": return self._parse_experiment_decl()
        elif kw == "evaluate":   return self._parse_eval_stmt()
        elif kw == "report":     return self._parse_report_stmt()
        else:
            raise ParseError(
                "a top-level declaration keyword "
                "('dataset', 'split', 'model', 'experiment', 'evaluate', 'report')",
                tok
            )

    # ══════════════════════════════════════════════════════════════════════════
    # DATASET
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_dataset_decl(self) -> DatasetDeclNode:
        """
        <dataset_decl> ::= "dataset" <identifier> "=" "load" "(" <string_literal> ")" ";"
        """
        self._consume_keyword("dataset")
        name = self._consume("IDENTIFIER").value
        self._consume("OP_ASSIGN")                            # =
        self._consume_keyword("load")
        self._consume("LPAREN")                               # (
        raw  = self._consume("STRING_LITERAL").value          # "path"
        self._consume("RPAREN")                               # )
        self._consume("SEMICOLON")                            # ;
        return DatasetDeclNode(name, self._strip_quotes(raw))

    def _parse_split_stmt(self) -> SplitStmtNode:
        """
        <split_stmt> ::= "split" <identifier> "into"
                         <split_part> "," <split_part> "," <split_part> ";"
        Exactly three parts are required by the grammar.
        """
        self._consume_keyword("split")
        dataset = self._consume("IDENTIFIER").value
        self._consume_keyword("into")

        parts = [self._parse_split_part()]
        self._consume("COMMA")
        parts.append(self._parse_split_part())
        self._consume("COMMA")
        parts.append(self._parse_split_part())

        self._consume("SEMICOLON")
        return SplitStmtNode(dataset, parts)

    def _parse_split_part(self) -> SplitPartNode:
        """
        <split_part> ::= <split_role> "(" <float_literal> ")"
        <split_role> ::= "train" | "validation" | "test"
        Split roles are emitted as KEYWORD tokens by the student's lexer.
        """
        tok = self._peek()
        if tok.type != "KEYWORD" or tok.value not in _SPLIT_ROLES:
            raise ParseError(
                "a split role ('train', 'validation', 'test')", tok
            )
        role = self._advance().value
        self._consume("LPAREN")
        ratio_tok = self._peek()
        if ratio_tok.type not in ("FLOAT_LITERAL", "INT_LITERAL"):
            raise ParseError("a float literal for the split ratio", ratio_tok)
        ratio = float(self._advance().value)
        self._consume("RPAREN")
        return SplitPartNode(role, ratio)

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_model_decl(self) -> ModelDeclNode:
        """
        <model_decl> ::= "model" <model_type> <identifier> "{" <field_list> "}"
        Model types are emitted as KEYWORD tokens.
        """
        self._consume_keyword("model")

        tok = self._peek()
        if tok.type != "KEYWORD" or tok.value not in _MODEL_TYPES:
            raise ParseError(
                f"a model type ({', '.join(sorted(_MODEL_TYPES))})", tok
            )
        model_type = self._advance().value

        name = self._consume("IDENTIFIER").value
        self._consume("LBRACE")
        fields = self._parse_field_list()
        self._consume("RBRACE")
        return ModelDeclNode(model_type, name, fields)

    def _parse_field_list(self) -> list:
        """
        <field_list> ::= <field> { "," <field> }
        """
        fields = [self._parse_field()]
        while self._is("COMMA"):
            self._advance()
            if self._is("RBRACE"):    # tolerate trailing comma
                break
            fields.append(self._parse_field())
        return fields

    def _parse_field(self) -> FieldNode:
        """
        <field>       ::= <identifier> "=" <field_value>
        <field_value> ::= <int_literal> | <float_literal>
                        | <string_literal> | <bool_literal>
        """
        key = self._consume("IDENTIFIER").value
        self._consume("OP_ASSIGN")
        value = self._parse_field_value()
        return FieldNode(key, value)

    def _parse_field_value(self) -> LiteralNode:
        """
        Dispatches on the token types the student's lexer emits:
          INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL, BOOL_LITERAL
        """
        tok = self._peek()
        if tok.type == "INT_LITERAL":
            return LiteralNode("int",    int(self._advance().value))
        elif tok.type == "FLOAT_LITERAL":
            return LiteralNode("float",  float(self._advance().value))
        elif tok.type == "STRING_LITERAL":
            return LiteralNode("string", self._strip_quotes(self._advance().value))
        elif tok.type == "BOOL_LITERAL":
            return LiteralNode("bool",   self._advance().value == "true")
        else:
            raise ParseError(
                "a field value (INT_LITERAL, FLOAT_LITERAL, "
                "STRING_LITERAL, or BOOL_LITERAL)", tok
            )

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_experiment_decl(self) -> ExperimentDeclNode:
        """
        <experiment_decl> ::= "experiment" <identifier> "{" { <experiment_stmt> } "}"
        """
        self._consume_keyword("experiment")
        name = self._consume("IDENTIFIER").value
        self._consume("LBRACE")

        stmts = []
        while not self._is("RBRACE"):
            if self._is("EOF"):
                raise ParseError("'}' to close experiment block", self._peek())
            stmts.append(self._parse_experiment_stmt())

        self._consume("RBRACE")
        return ExperimentDeclNode(name, stmts)

    def _parse_experiment_stmt(self) -> ASTNode:
        """
        Predictive dispatch on the current keyword.
        """
        tok = self._peek()
        kw  = tok.value
        if   kw == "train":    return self._parse_train_stmt()
        elif kw == "evaluate": return self._parse_evaluate_stmt()
        elif kw == "collect":  return self._parse_collect_stmt()
        elif kw == "analyze":  return self._parse_analyze_stmt()
        elif kw == "compare":  return self._parse_compare_stmt()
        elif kw == "exclude":  return self._parse_exclude_stmt()
        elif kw == "select":   return self._parse_select_stmt()
        else:
            raise ParseError(
                "an experiment statement keyword "
                "('train','evaluate','collect','analyze',"
                "'compare','exclude','select')",
                tok
            )

    def _parse_train_stmt(self) -> TrainStmtNode:
        """train <identifier> on <dataset_ref>;"""
        self._consume_keyword("train")
        model = self._consume("IDENTIFIER").value
        self._consume_keyword("on")
        dref  = self._parse_dataset_ref()
        self._consume("SEMICOLON")
        return TrainStmtNode(model, dref)

    def _parse_evaluate_stmt(self) -> EvaluateStmtNode:
        """evaluate <identifier> on <dataset_ref>;   (inside experiment)"""
        self._consume_keyword("evaluate")
        model = self._consume("IDENTIFIER").value
        self._consume_keyword("on")
        dref  = self._parse_dataset_ref()
        self._consume("SEMICOLON")
        return EvaluateStmtNode(model, dref)

    def _parse_dataset_ref(self) -> DatasetRefNode:
        """
        <dataset_ref> ::= <identifier> "." <split_role>
        The dataset name is always an IDENTIFIER.
        The split role is a KEYWORD ("train" | "validation" | "test").
        """
        dataset = self._consume("IDENTIFIER").value
        self._consume("DOT")
        tok = self._peek()
        if tok.type != "KEYWORD" or tok.value not in _SPLIT_ROLES:
            raise ParseError(
                f"a split role after '.' ({', '.join(sorted(_SPLIT_ROLES))})", tok
            )
        role = self._advance().value
        return DatasetRefNode(dataset, role)

    def _parse_collect_stmt(self) -> CollectStmtNode:
        """collect metrics [ <metric_list> ];"""
        self._consume_keyword("collect")
        self._consume_keyword("metrics")
        self._consume("LBRACKET")
        metrics = self._parse_metric_list()
        self._consume("RBRACKET")
        self._consume("SEMICOLON")
        return CollectStmtNode(metrics)

    def _parse_metric_list(self) -> list:
        """
        <metric_list> ::= <metric_literal> { "," <metric_literal> }
        Metrics are emitted as METRIC_LITERAL tokens by the student's lexer.
        """
        metrics = [self._parse_metric_literal()]
        while self._is("COMMA"):
            self._advance()
            metrics.append(self._parse_metric_literal())
        return metrics

    def _parse_metric_literal(self) -> str:
        """
        Accepts a METRIC_LITERAL token.
        The student's lexer classifies accuracy/precision/recall/f1 as
        METRIC_LITERAL  (NOT KEYWORD or IDENTIFIER).
        """
        tok = self._peek()
        if tok.type != "METRIC_LITERAL":
            raise ParseError(
                "a metric literal (METRIC_LITERAL: "
                "accuracy, precision, recall, f1)", tok
            )
        return self._advance().value

    def _parse_analyze_stmt(self) -> AnalyzeStmtNode:
        """
        <analyze_stmt>       ::= "analyze" "overfitting" "{" <analyze_field_list> "}"
        <analyze_field_list> ::= <analyze_field> { <analyze_field> }

        D1 CHANGE: fields are NOT comma-separated.
        Each field is self-terminating with ";", so the list is simply
        zero-or-more repetitions of <analyze_field> until "}" is seen.

        D1 sample program:
            analyze overfitting
            {
                threshold = 0.10;
                underfit_max_acc = 0.70;
            }
        No comma appears between the two fields.
        """
        self._consume_keyword("analyze")
        self._consume_keyword("overfitting")
        self._consume("LBRACE")

        fields = []
        # Loop: keep parsing analyze_fields as long as the next token is a
        # recognised field keyword, stopping when "}" or EOF is reached.
        while self._peek().type == "KEYWORD" and               self._peek().value in ("threshold", "underfit_max_acc"):
            fields.append(self._parse_analyze_field())

        if not fields:
            raise ParseError(
                "at least one analyze field (KEYWORD 'threshold' or "
                "'underfit_max_acc')", self._peek()
            )

        self._consume("RBRACE")
        return AnalyzeStmtNode(fields)

    def _parse_analyze_field(self) -> AnalyzeFieldNode:
        """
        "threshold" "=" <float> ";"  |  "underfit_max_acc" "=" <float> ";"
        Both names are KEYWORD tokens.
        """
        tok = self._peek()
        if tok.type != "KEYWORD" or tok.value not in ("threshold", "underfit_max_acc"):
            raise ParseError(
                "analyze field name (KEYWORD 'threshold' or 'underfit_max_acc')", tok
            )
        key = self._advance().value
        self._consume("OP_ASSIGN")
        val_tok = self._peek()
        if val_tok.type not in ("FLOAT_LITERAL", "INT_LITERAL"):
            raise ParseError(
                "a numeric literal (FLOAT_LITERAL or INT_LITERAL) "
                "for the analyze field value", val_tok
            )
        value = float(self._advance().value)
        self._consume("SEMICOLON")
        return AnalyzeFieldNode(key, value)

    def _parse_compare_stmt(self) -> CompareStmtNode:
        """compare by <metric> <direction>;"""
        self._consume_keyword("compare")
        self._consume_keyword("by")
        metric = self._parse_metric_literal()
        tok = self._peek()
        if tok.type != "KEYWORD" or tok.value not in _COMPARE_DIRS:
            raise ParseError(
                f"a comparison direction KEYWORD "
                f"({', '.join(sorted(_COMPARE_DIRS))})", tok
            )
        direction = self._advance().value
        self._consume("SEMICOLON")
        return CompareStmtNode(metric, direction)

    def _parse_exclude_stmt(self) -> ExcludeStmtNode:
        """exclude if <exclude_condition>;"""
        self._consume_keyword("exclude")
        self._consume_keyword("if")

        conditions = [self._parse_exclude_term()]
        while self._is("KEYWORD", "or"):
            self._advance()
            conditions.append(self._parse_exclude_term())

        self._consume("SEMICOLON")
        return ExcludeStmtNode(conditions)

    def _parse_exclude_term(self) -> str:
        tok = self._peek()
        if tok.type != "KEYWORD" or tok.value not in ("overfit", "underfit"):
            raise ParseError("KEYWORD 'overfit' or 'underfit'", tok)
        return self._advance().value

    def _parse_select_stmt(self) -> SelectStmtNode:
        self._consume_keyword("select")
        self._consume_keyword("best")
        self._consume("SEMICOLON")
        return SelectStmtNode()

    # ══════════════════════════════════════════════════════════════════════════
    # TOP-LEVEL EVALUATE
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_eval_stmt(self) -> EvalStmtNode:
        """
        <eval_stmt>   ::= "evaluate" <eval_target> "on" <dataset_ref> ";"
        <eval_target> ::= <identifier>
                        | "[" <identifier> "." <identifier> "]"
        """
        self._consume_keyword("evaluate")
        target = self._parse_eval_target()
        self._consume_keyword("on")
        dref = self._parse_dataset_ref()
        self._consume("SEMICOLON")
        return EvalStmtNode(target, dref)

    def _parse_eval_target(self) -> ASTNode:
        """
        <eval_target> ::= <identifier>
                        | <identifier> "." <identifier>      ← D1 sample form
                        | "[" <identifier> "." <identifier> "]"

        D1 CHANGE: the simple (no-bracket) branch now also accepts a dotted
        access.  The D1 sample program uses:
            evaluate comparison.best on iris.test;
        "comparison.best" is two plain identifiers joined by ".", with NO
        brackets.  The EBNF only lists <identifier> for the simple branch, but
        the authoritative sample shows this dotted form is valid.

        Disambiguation: we parse the first IDENTIFIER, then peek at the next
        token.  If it is a DOT we consume it and the following name; otherwise
        we return a plain identifier target.

        Bracket target uses LBRACKET as its unambiguous lead token.
        After the DOT we accept both IDENTIFIER and KEYWORD because "best"
        is classified as KEYWORD by the student's lexer.
        """
        if self._is("LBRACKET"):
            self._advance()                                   # consume "["
            obj = self._consume("IDENTIFIER").value
            self._consume("DOT")
            attr_tok = self._peek()
            if attr_tok.type not in ("IDENTIFIER", "KEYWORD"):
                raise ParseError(
                    "an IDENTIFIER or KEYWORD attribute name after '.'", attr_tok
                )
            attr = self._advance().value
            self._consume("RBRACKET")
            return EvalTargetBracketNode(obj, attr)
        else:
            # Simple branch: <identifier> [ "." <identifier> ]
            name = self._consume("IDENTIFIER").value
            if self._is("DOT"):
                self._advance()                               # consume "."
                attr_tok = self._peek()
                if attr_tok.type not in ("IDENTIFIER", "KEYWORD"):
                    raise ParseError(
                        "an IDENTIFIER or KEYWORD after '.' in eval target",
                        attr_tok
                    )
                attr = self._advance().value
                return EvalTargetSimpleNode(name, attr)
            return EvalTargetSimpleNode(name)

    # ══════════════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_report_stmt(self) -> ReportStmtNode:
        """report <identifier> { <report_field_list> }"""
        self._consume_keyword("report")
        name = self._consume("IDENTIFIER").value
        self._consume("LBRACE")

        fields = [self._parse_report_field()]
        while self._is("COMMA"):
            self._advance()
            if self._is("RBRACE"):
                break
            fields.append(self._parse_report_field())

        self._consume("RBRACE")
        return ReportStmtNode(name, fields)

    def _parse_report_field(self) -> ASTNode:
        """
        "metrics" "=" "[" <metric_list> "]"
        | "show" "=" <show_value>
        Both "metrics" and "show" are KEYWORD tokens.
        """
        tok = self._peek()
        if tok.type == "KEYWORD" and tok.value == "metrics":
            self._advance()
            self._consume("OP_ASSIGN")
            self._consume("LBRACKET")
            metrics = self._parse_metric_list()
            self._consume("RBRACKET")
            return ReportFieldMetricsNode(metrics)
        elif tok.type == "KEYWORD" and tok.value == "show":
            self._advance()
            self._consume("OP_ASSIGN")
            val_tok = self._peek()
            if val_tok.type == "KEYWORD" and val_tok.value == "overfitting_analysis":
                value = self._advance().value
            elif val_tok.type == "STRING_LITERAL":
                value = self._strip_quotes(self._advance().value)
            else:
                raise ParseError(
                    "KEYWORD 'overfitting_analysis' or a STRING_LITERAL "
                    "for 'show'", val_tok
                )
            return ReportFieldShowNode(value)
        else:
            raise ParseError(
                "KEYWORD 'metrics' or KEYWORD 'show' for a report field", tok
            )

    # ══════════════════════════════════════════════════════════════════════════
    # EXPRESSIONS
    # (operator precedence encoded in the call hierarchy – Sebesta §3.6 / §4.4)
    # ══════════════════════════════════════════════════════════════════════════

    def parse_expr(self) -> ASTNode:
        """
        <expr> ::= <and_expr> { "or" <and_expr> }
        Left-associative iteration.
        """
        node = self._parse_and_expr()
        while self._is("KEYWORD", "or"):
            self._advance()
            node = BinOpNode("or", node, self._parse_and_expr())
        return node

    def _parse_and_expr(self) -> ASTNode:
        """<and_expr> ::= <not_expr> { "and" <not_expr> }"""
        node = self._parse_not_expr()
        while self._is("KEYWORD", "and"):
            self._advance()
            node = BinOpNode("and", node, self._parse_not_expr())
        return node

    def _parse_not_expr(self) -> ASTNode:
        """
        <not_expr> ::= <rel_expr> | "not" <not_expr>
        Right-recursive so "not not x" = "not (not x)".
        """
        if self._is("KEYWORD", "not"):
            self._advance()
            return UnaryOpNode("not", self._parse_not_expr())
        return self._parse_rel_expr()

    def _parse_rel_expr(self) -> ASTNode:
        """
        <rel_expr> ::= <add_expr> [ <rel_op> <add_expr> ]
        Relational operators are non-associative (single optional comparison).
        """
        left = self._parse_add_expr()
        if self._peek().type in _REL_OP_TYPES:
            op    = self._advance().value
            right = self._parse_add_expr()
            return BinOpNode(op, left, right)
        return left

    def _parse_add_expr(self) -> ASTNode:
        """
        <add_expr> ::= <mul_expr> { ("+" | "-") <mul_expr> }

        LEFT-ASSOCIATIVITY MECHANISM (Sebesta §4.4):
        The while-loop wraps the accumulated left subtree as the NEW left
        child on every iteration, producing a left-leaning tree.
        """
        node = self._parse_mul_expr()
        while self._peek().type in ("OP_PLUS", "OP_MINUS"):
            op   = self._advance().value
            node = BinOpNode(op, node, self._parse_mul_expr())
        return node

    def _parse_mul_expr(self) -> ASTNode:
        """
        <mul_expr> ::= <unary_expr> { ("*" | "/") <unary_expr> }
        Left-associative (same iterative pattern).
        """
        node = self._parse_unary_expr()
        while self._peek().type in ("OP_MUL", "OP_DIV"):
            op   = self._advance().value
            node = BinOpNode(op, node, self._parse_unary_expr())
        return node

    def _parse_unary_expr(self) -> ASTNode:
        """
        <unary_expr> ::= <primary> | "-" <unary_expr>
        Right-recursive so  --x = -(-(x)).
        """
        if self._is("OP_MINUS"):
            self._advance()
            return UnaryOpNode("-", self._parse_unary_expr())
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        """
        <primary> ::= INT_LITERAL | FLOAT_LITERAL | BOOL_LITERAL
                    | STRING_LITERAL | METRIC_LITERAL
                    | IDENTIFIER [ "." IDENTIFIER ]
                    | "(" <expr> ")"

        Base case of the expression recursion.
        """
        tok = self._peek()

        if tok.type == "INT_LITERAL":
            return LiteralNode("int",    int(self._advance().value))
        if tok.type == "FLOAT_LITERAL":
            return LiteralNode("float",  float(self._advance().value))
        if tok.type == "STRING_LITERAL":
            return LiteralNode("string", self._strip_quotes(self._advance().value))
        if tok.type == "BOOL_LITERAL":
            return LiteralNode("bool",   self._advance().value == "true")
        if tok.type == "METRIC_LITERAL":
            return LiteralNode("metric", self._advance().value)
        if self._is("LPAREN"):
            self._advance()
            inner = self.parse_expr()
            self._consume("RPAREN")
            return inner
        if tok.type == "IDENTIFIER":
            name = self._advance().value
            if self._is("DOT"):
                self._advance()
                attr = self._consume("IDENTIFIER").value
                return AttributeAccessNode(name, attr)
            return IdentifierNode(name)

        raise ParseError(
            "a primary expression (literal, identifier, or parenthesised expression)",
            tok
        )

    # ══════════════════════════════════════════════════════════════════════════
    # AST DUMP
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def dump_ast(node: ASTNode) -> str:
        return node.dump(indent=0)

    @staticmethod
    def print_tree(node: ASTNode) -> None:
        print(Parser.dump_ast(node))


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST  (python3 ayar_parser.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ayar_lexer import Lexer

    # ── D1 sample program (from Design Specification Document) ───────────────
    SRC = r'''
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
'''
    tokens = Lexer(SRC).tokenize()
    ast    = Parser(tokens).parse()
    Parser.print_tree(ast)
