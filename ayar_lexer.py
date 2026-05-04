"""
ayar_lexer.py
─────────────────────────────────────────────────────────────────────────────
Lexical Analyser (Scanner) for the "ayar" DSL.

Design reference: Sebesta – "Concepts of Programming Languages" 12th ed., Ch. 4
  • Ch. 4.1  – Introduction to Lexical Analysis
  • Ch. 4.2  – The Parsing Problem  (tokens as the alphabet for the parser)
  • Ch. 4.3  – Recursive-Descent Parsing foundation (lexer feeds the parser)

Tokenisation strategy: single-pass, regex-driven, object-oriented.
Each token is a dataclass that carries type, value, line, and column so the
parser (and error messages) always have full source-location information.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOKEN DATACLASS
#    (Sebesta §4.1 – a lexeme is the string matched; a token is the categorised
#    unit the lexer hands to the parser.)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Token:
    """
    Represents a single token produced by the lexer.

    Fields
    ------
    type         : str  – token category  (e.g. 'KEYWORD', 'IDENTIFIER', …)
    value        : str  – the exact source text that was matched (the lexeme)
    line_number  : int  – 1-based source line where the token starts
    column_number: int  – 1-based column index where the token starts
    """
    type: str
    value: str
    line_number: int
    column_number: int

    def __repr__(self) -> str:
        return (
            f"Token(type={self.type!r}, value={self.value!r}, "
            f"line={self.line_number}, col={self.column_number})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOM EXCEPTION
#    (Sebesta §4.1 – lexical errors must report the offending character and its
#    location so the programmer can find and fix the mistake quickly.)
# ─────────────────────────────────────────────────────────────────────────────

class LexicalError(Exception):
    """
    Raised when the lexer encounters a character (or character sequence) that
    does not match any known lexical pattern.

    Parameters
    ----------
    char        : the illegal character
    line_number : source line   (1-based)
    col_number  : source column (1-based)
    """
    def __init__(self, char: str, line_number: int, col_number: int):
        self.char = char
        self.line_number = line_number
        self.col_number = col_number
        super().__init__(
            f"LexicalError: illegal character {char!r} "
            f"at line {line_number}, column {col_number}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. KEYWORD & SPECIAL-WORD TABLES
#    (Sebesta §3.3 – reserved words are the finite vocabulary of the language;
#    they must be recognised *after* an identifier-shaped lexeme is matched,
#    not before, so that names like "trainer" are not mis-tokenised.)
# ─────────────────────────────────────────────────────────────────────────────

# Every word that has a fixed meaning in ayar grammar.
# Stored as a frozenset for O(1) membership testing.
KEYWORDS: frozenset = frozenset({
    # dataset / split
    "dataset", "load", "split", "into",
    # split roles  (also valid in dataset_ref)
    "train", "validation", "test",
    # model
    "model",
    # model types  (treated as keywords – they appear in a fixed syntactic slot)
    "KNN", "DecisionTree", "SVM", "NaiveBayes", "LogisticRegression",
    # experiment
    "experiment", "on",
    # experiment statements
    "evaluate", "collect", "metrics",
    "analyze", "overfitting",
    "threshold", "underfit_max_acc",
    "compare", "by", "higher_is_better", "lower_is_better",
    "exclude", "if", "overfit", "underfit",
    "select", "best",
    # report
    "report", "show", "overfitting_analysis",
    # logical operators  (keyword form, not symbols)
    "and", "or", "not",
})

# Metric literals are a distinct token type (METRIC_LITERAL) separate from
# both KEYWORD and IDENTIFIER, because the grammar treats them as typed values
# that can appear inside metric lists and expressions.
METRIC_LITERALS: frozenset = frozenset({"accuracy", "precision", "recall", "f1"})

# Boolean literals similarly form their own token type.
BOOL_LITERALS: frozenset = frozenset({"true", "false"})


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOKEN SPECIFICATION  (ordered list of (type, compiled_regex) pairs)
#    (Sebesta §4.2 – a scanner is essentially an implementation of a finite
#    automaton whose transition function is described by regular expressions.)
#
#    Order matters:
#      • Multi-character operators (==, !=, <=, >=) must come BEFORE their
#        single-character prefixes (=, !, <, >).
#      • FLOAT must come before INT so "3.14" is not split into INT "3" and
#        something beginning with ".".
#      • STRING must be tried early to prevent its content from being
#        tokenised as other types.
#      • COMMENT and WHITESPACE are handled separately (they are consumed but
#        not appended to the token list).
# ─────────────────────────────────────────────────────────────────────────────

# Each tuple is (token_type_string, raw_regex_pattern).
# re.compile() is called once at module load for efficiency.
TOKEN_SPEC: List[tuple] = [

    # ── whitespace & comments (consumed, not yielded) ──
    # NEWLINE is tracked explicitly so the lexer can increment line_number.
    ("NEWLINE",         r"\n"),
    ("WHITESPACE",      r"[ \t\r]+"),
    ("COMMENT",         r"//[^\n]*"),          # rest of line after //

    # ── string literal ──
    # Matches an opening quote, any number of non-quote / non-bare-newline
    # characters (escape sequences such as \" are allowed), then closing quote.
    ("STRING_LITERAL",  r'"(?:[^"\\\n]|\\.)*"'),

    # ── numeric literals ──
    # FLOAT before INT; two legal forms:
    #   digit+ . digit*      →  3.   3.14
    #   . digit+             →  .5
    ("FLOAT_LITERAL",   r'\d+\.\d*|\.\d+'),
    ("INT_LITERAL",     r'\d+'),

    # ── identifier / keyword-shaped tokens ──
    # A single pattern captures every letter/digit/underscore run.
    # The lexer classifies it afterwards (keyword vs bool vs metric vs ident).
    ("WORD",            r'[A-Za-z_][A-Za-z0-9_]*'),

    # ── two-character operators (must precede single-char forms) ──
    ("OP_EQ",           r'=='),
    ("OP_NEQ",          r'!='),
    ("OP_LTE",          r'<='),
    ("OP_GTE",          r'>='),

    # ── single-character operators & punctuation ──
    ("OP_ASSIGN",       r'='),
    ("OP_LT",           r'<'),
    ("OP_GT",           r'>'),
    ("OP_PLUS",         r'\+'),
    ("OP_MINUS",        r'-'),
    ("OP_MUL",          r'\*'),
    ("OP_DIV",          r'/'),

    # ── separators ──
    ("LBRACE",          r'\{'),
    ("RBRACE",          r'\}'),
    ("LPAREN",          r'\('),
    ("RPAREN",          r'\)'),
    ("LBRACKET",        r'\['),
    ("RBRACKET",        r'\]'),
    ("COMMA",           r','),
    ("SEMICOLON",       r';'),
    ("DOT",             r'\.'),
]

# Build a single combined regex using named groups for each pattern.
# The alternation is ordered (Python re tries alternatives left-to-right),
# which preserves the priority ordering described above.
_MASTER_PATTERN: re.Pattern = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC)
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. LEXER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Lexer:
    """
    Converts an ayar source string into a flat list of Token objects.

    Usage
    -----
        lexer  = Lexer(source_code)
        tokens = lexer.tokenize()

    Raises
    ------
        LexicalError  – on any character that does not belong to the language
                        alphabet (Sebesta §4.1).
    """

    def __init__(self, source: str):
        """
        Parameters
        ----------
        source : the complete ayar source text to lex.
        """
        self._source: str = source

    # ── public interface ──────────────────────────────────────────────────────

    def tokenize(self) -> List[Token]:
        """
        Scan the entire source text and return a list of Token objects.

        Algorithm (Sebesta §4.2 – maximal munch / longest match):
          The combined regex is scanned left-to-right.  At each position the
          engine tries all alternatives simultaneously (because they are
          joined by |) and returns the *first* (highest-priority) match.
          Because Python's re guarantees the leftmost match at each position,
          and we have ordered the spec so that longer patterns precede shorter
          ones, we get the standard "longest-match wins" behaviour described
          by Sebesta.

        Line & column tracking:
          We maintain `line` and `col` counters manually:
            • A NEWLINE match increments `line` and resets `col` to 1.
            • Every other match advances `col` by the length of the matched
              lexeme.
            • Skipped characters (whitespace, comments) still advance `col`.
        """
        tokens: List[Token] = []
        line: int = 1         # current 1-based line number
        col: int  = 1         # current 1-based column number
        pos: int  = 0         # current character offset in the source string

        while pos < len(self._source):

            # ── try to match at the current position ──────────────────────
            match = _MASTER_PATTERN.match(self._source, pos)

            if match is None:
                # No pattern matched → illegal character.
                raise LexicalError(self._source[pos], line, col)

            token_type: str = match.lastgroup   # name of the matched group
            lexeme: str     = match.group()     # the matched text
            token_col: int  = col               # column where this token starts

            # ── advance position & update line/column counters ────────────
            pos += len(lexeme)

            if token_type == "NEWLINE":
                line += 1
                col   = 1
                # Newlines are not yielded as tokens.
                continue

            if token_type in ("WHITESPACE", "COMMENT"):
                # Advance column but do not yield a token.
                col += len(lexeme)
                continue

            # All other token types advance the column.
            col += len(lexeme)

            # ── classify WORD tokens ──────────────────────────────────────
            # A WORD matched the identifier-shaped pattern; we now decide its
            # actual token type by checking the fixed-word tables.
            # (Sebesta §3.3 – reserved words are recognised as a post-scan step.)
            if token_type == "WORD":
                if lexeme in BOOL_LITERALS:
                    token_type = "BOOL_LITERAL"
                elif lexeme in METRIC_LITERALS:
                    token_type = "METRIC_LITERAL"
                elif lexeme in KEYWORDS:
                    token_type = "KEYWORD"
                else:
                    token_type = "IDENTIFIER"

            # ── append the fully classified token ─────────────────────────
            tokens.append(Token(
                type          = token_type,
                value         = lexeme,
                line_number   = line,
                column_number = token_col,
            ))

        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 6. SELF-TEST  –  Sample ayar Program
#    Run this file directly:  python ayar_lexer.py
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_PROGRAM = r"""
// ── Dataset loading & splitting ──
dataset iris = load("iris.csv");

split iris into
    train(0.7),
    validation(0.15),
    test(0.15);

// ── Model declarations ──
model KNN knn_model {
    n_neighbors = 5,
    weights = "uniform"
}

model DecisionTree dt_model {
    max_depth = 10,
    criterion = "gini"
}

model SVM svm_model {
    C = 1.0,
    kernel = "rbf",
    probability = true
}

model NaiveBayes nb_model {
    var_smoothing = 1e-9
}

model LogisticRegression lr_model {
    C = 0.5,
    max_iter = 200,
    solver = "lbfgs"
}

// ── Experiment ──
experiment main_exp {
    train knn_model on iris.train;
    train dt_model  on iris.train;
    train svm_model on iris.train;
    train nb_model  on iris.train;
    train lr_model  on iris.train;

    evaluate knn_model on iris.validation;
    evaluate dt_model  on iris.validation;
    evaluate svm_model on iris.validation;
    evaluate nb_model  on iris.validation;
    evaluate lr_model  on iris.validation;

    collect metrics [accuracy, precision, recall, f1];

    analyze overfitting {
        threshold = 0.05;,
        underfit_max_acc = 0.6;
    }

    compare by accuracy higher_is_better;

    exclude if overfit or underfit;

    select best;
}

// ── Final evaluation ──
evaluate [main_exp.best] on iris.test;

// ── Report ──
report final_report {
    metrics = [accuracy, precision, recall, f1],
    show    = overfitting_analysis
}
"""


def main() -> None:
    """Lex the sample program and pretty-print every token."""
    lexer = Lexer(SAMPLE_PROGRAM)

    try:
        tokens = lexer.tokenize()
    except LexicalError as err:
        print(err)
        return

    # Column widths for aligned output
    w_type  = max(len(t.type)  for t in tokens) + 2
    w_value = max(len(t.value) for t in tokens) + 2

    header = (
        f"{'TYPE':<{w_type}} "
        f"{'VALUE':<{w_value}} "
        f"{'LINE':>5}  "
        f"{'COL':>5}"
    )
    print(header)
    print("─" * len(header))

    for tok in tokens:
        print(
            f"{tok.type:<{w_type}} "
            f"{tok.value!r:<{w_value}} "
            f"{tok.line_number:>5}  "
            f"{tok.column_number:>5}"
        )

    print(f"\n✓  {len(tokens)} tokens produced.")


if __name__ == "__main__":
    main()
