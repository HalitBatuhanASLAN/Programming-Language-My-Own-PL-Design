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

Changes from previous version (aligned with D1 Design Specification):
  FIX-1  IDENTIFIER pattern now requires a LETTER as the first character,
         matching the EBNF rule exactly:
           <identifier> ::= <letter> { <letter> | <digit> | "_" }
         The old pattern r'[A-Za-z_][A-Za-z0-9_]*' also accepted a leading
         underscore, which the grammar does not allow.

  FIX-2  Scientific-notation numbers (e.g. 1e-9) are NOT part of the language.
         The EBNF only defines:
           <float_literal> ::= <digit>{ <digit> } "." { <digit> }
                             | "." <digit> { <digit> }
         No exponent form is specified.  The old sample program contained
         "var_smoothing = 1e-9" which is therefore not legal ayar source.

  FIX-3  The sample program now reproduces the exact example from D1 §"Language
         Overview", which:
           • Uses only the two models declared there (KNN, DecisionTree).
           • Uses the dotted eval_target form  evaluate iris_exp.best on iris.test;
             (not the bracketed form, which is a separate valid alternative).
           • Uses NO comma between analyze-overfitting fields — the EBNF rule
               <analyze_field_list> ::= <analyze_field> { <analyze_field> }
             has no "," separator; each field is terminated by its own ";".

  FIX-4  A second sample program (SAMPLE_BRACKET_EVAL) is added to demonstrate
         the bracketed eval_target form  evaluate [iris_exp.best] on iris.test;
         so all three variants of eval_target are exercised and visible.
"""

import re
from dataclasses import dataclass
from typing import List


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

# Every word that has a fixed meaning in the ayar grammar.
# Stored as a frozenset for O(1) average-case membership testing.
KEYWORDS: frozenset = frozenset({
    # ── dataset / split ──────────────────────────────────────────────────────
    "dataset", "load", "split", "into",

    # ── split roles  (also used as dataset_ref suffixes) ─────────────────────
    "train", "validation", "test",

    # ── model declaration ────────────────────────────────────────────────────
    "model",

    # ── model types  (appear in a fixed syntactic position after "model") ────
    "KNN", "DecisionTree", "SVM", "NaiveBayes", "LogisticRegression",

    # ── experiment ───────────────────────────────────────────────────────────
    "experiment", "on",

    # ── experiment statements ─────────────────────────────────────────────────
    "evaluate", "collect", "metrics",
    "analyze",  "overfitting",
    "threshold", "underfit_max_acc",
    "compare",  "by", "higher_is_better", "lower_is_better",
    "exclude",  "if", "overfit", "underfit",
    "select",   "best",

    # ── report ───────────────────────────────────────────────────────────────
    "report", "show", "overfitting_analysis",

    # ── logical operators (keyword form, not symbol form) ────────────────────
    "and", "or", "not",
})

# Metric literals form their own token type (METRIC_LITERAL).
# The grammar treats them as typed values that can appear in metric lists and
# expressions, distinct from both keywords and plain identifiers.
METRIC_LITERALS: frozenset = frozenset({"accuracy", "precision", "recall", "f1"})

# Boolean literals also have their own token type (BOOL_LITERAL).
BOOL_LITERALS: frozenset = frozenset({"true", "false"})


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOKEN SPECIFICATION  (ordered list of (type, raw_pattern) pairs)
#    (Sebesta §4.2 – a scanner implements a finite automaton whose transition
#    function is described by regular expressions.)
#
#    ORDER IS CRITICAL:
#      • NEWLINE before WHITESPACE so '\n' increments the line counter rather
#        than just advancing the column.
#      • BLOCK_COMMENT (/* … */) before COMMENT (//…) and before OP_DIV (/)
#        so that "/*" is never split into two tokens.
#      • COMMENT (//…) before OP_DIV (/) so "//" is never split into two
#        division operators.
#      • STRING_LITERAL early so its content is never re-tokenised.
#      • FLOAT_LITERAL before INT_LITERAL — maximal munch: "3.14" must be one
#        FLOAT token, not INT "3" followed by something starting with ".".
#      • WORD before operators (it cannot start with a non-letter anyway, but
#        explicit ordering documents intent).
#      • Two-character operators (==, !=, <=, >=) BEFORE their single-char
#        prefixes (=, !, <, >) for the same maximal-munch reason.
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_SPEC: List[tuple] = [

    # ── whitespace & comments (consumed, never yielded as tokens) ────────────
    # NEWLINE is a distinct entry so the main loop can increment line_number.
    ("NEWLINE",        r"\n"),
    ("WHITESPACE",     r"[ \t\r]+"),
    # BLOCK_COMMENT must precede COMMENT (which starts with /) and OP_DIV (/)
    # so that "/*" is never split into OP_DIV + OP_MUL.
    # [\s\S]*? matches any character including newlines (non-greedy).
    # BLOCK_COMMENT is silently consumed exactly like COMMENT.
    ("BLOCK_COMMENT",  r"/\*[\s\S]*?\*/"),
    ("COMMENT",        r"//[^\n]*"),          # everything from // to end-of-line

    # ── string literal ────────────────────────────────────────────────────────
    # EBNF: <string_literal> ::= '"' { <string_char> } '"'
    # <string_char> is any character except '"' and an unescaped newline.
    # The pattern also allows \" escape sequences inside the string.
    ("STRING_LITERAL", r'"(?:[^"\\\n]|\\.)*"'),

    # ── numeric literals ──────────────────────────────────────────────────────
    # EBNF:
    #   <float_literal> ::= <digit> { <digit> } "." { <digit> }
    #                     | "." <digit> { <digit> }
    #   <int_literal>   ::= <digit> { <digit> }
    #
    # NOTE: Scientific notation (e.g. 1e-9) is NOT in the grammar.  Any source
    # text that looks like a float in that form will be lexed as separate tokens
    # (INT "1", IDENTIFIER "e", OP_MINUS "-", INT "9") and the parser will
    # reject it — which is the correct behaviour per the specification.
    #
    # FLOAT must precede INT so "3.14" matches the longer pattern.
    ("FLOAT_LITERAL",  r'\d+\.\d*|\.\d+'),
    ("INT_LITERAL",    r'\d+'),

    # ── identifier / keyword-shaped tokens ────────────────────────────────────
    # EBNF: <identifier> ::= <letter> { <letter> | <digit> | "_" }
    #
    # FIX-1: the pattern MUST start with [A-Za-z], NOT [A-Za-z_].
    # A leading underscore is NOT a valid start for an identifier in ayar.
    # The post-match classification step below decides the final token type
    # (KEYWORD / BOOL_LITERAL / METRIC_LITERAL / IDENTIFIER).
    ("WORD",           r'[A-Za-z][A-Za-z0-9_]*'),

    # ── two-character operators (must precede single-char forms) ──────────────
    ("OP_EQ",          r'=='),
    ("OP_NEQ",         r'!='),
    ("OP_LTE",         r'<='),
    ("OP_GTE",         r'>='),

    # ── single-character operators ────────────────────────────────────────────
    ("OP_ASSIGN",      r'='),
    ("OP_LT",          r'<'),
    ("OP_GT",          r'>'),
    ("OP_PLUS",        r'\+'),
    ("OP_MINUS",       r'-'),
    ("OP_MUL",         r'\*'),
    ("OP_DIV",         r'/'),

    # ── separators / punctuation ──────────────────────────────────────────────
    ("LBRACE",         r'\{'),
    ("RBRACE",         r'\}'),
    ("LPAREN",         r'\('),
    ("RPAREN",         r'\)'),
    ("LBRACKET",       r'\['),
    ("RBRACKET",       r'\]'),
    ("COMMA",          r','),
    ("SEMICOLON",      r';'),
    ("DOT",            r'\.'),
]

# Compile all patterns into a single master regex using named capturing groups.
# Python's re engine tries alternatives left-to-right within the alternation,
# so the ordering of TOKEN_SPEC directly controls priority (Sebesta §4.2).
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
        LexicalError – on any character that does not belong to the language
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
          The combined regex is applied at each position.  Because all
          alternatives are tried simultaneously (joined by |), and longer
          patterns are listed before shorter ones, the engine always returns
          the longest possible match at each position — the standard
          "maximal munch" rule described by Sebesta.

        Line & column tracking:
          • A NEWLINE token increments `line` and resets `col` to 1.
          • WHITESPACE and COMMENT tokens advance `col` only (not yielded).
          • All other tokens advance `col` by their lexeme length.
          • `token_col` captures the column *before* advancing, so each Token
            records where it *starts*, not where it ends.
        """
        tokens: List[Token] = []
        line: int = 1    # current 1-based line number
        col:  int = 1    # current 1-based column number
        pos:  int = 0    # current byte offset in the source string

        while pos < len(self._source):

            # Try to match at the current position.
            match = _MASTER_PATTERN.match(self._source, pos)

            if match is None:
                # No pattern matched → character is not in the language alphabet.
                raise LexicalError(self._source[pos], line, col)

            token_type: str = match.lastgroup   # name of the winning group
            lexeme:     str = match.group()     # the matched source text
            token_col:  int = col               # column where this token starts

            pos += len(lexeme)                  # advance the source pointer

            # ── handle skipped token types ────────────────────────────────
            if token_type == "NEWLINE":
                line += 1
                col   = 1
                continue                        # do not append a token

            if token_type in ("WHITESPACE", "COMMENT"):
                col += len(lexeme)
                continue                        # do not append a token

            if token_type == "BLOCK_COMMENT":
                # A block comment can span multiple lines.
                # Count embedded newlines and update line/col accordingly.
                newline_count = lexeme.count("\n")
                if newline_count:
                    line += newline_count
                    # col resets to 1 + characters after the last newline
                    col = len(lexeme) - lexeme.rfind("\n")
                else:
                    col += len(lexeme)
                continue                        # do not append a token

            # ── advance column for all yielded tokens ─────────────────────
            col += len(lexeme)

            # ── post-match classification for WORD tokens ─────────────────
            # (Sebesta §3.3 – reserved words are recognised after the general
            # identifier pattern matches, not before.)
            if token_type == "WORD":
                if lexeme in BOOL_LITERALS:
                    token_type = "BOOL_LITERAL"
                elif lexeme in METRIC_LITERALS:
                    token_type = "METRIC_LITERAL"
                elif lexeme in KEYWORDS:
                    token_type = "KEYWORD"
                else:
                    token_type = "IDENTIFIER"

            tokens.append(Token(
                type          = token_type,
                value         = lexeme,
                line_number   = line,
                column_number = token_col,
            ))

        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAMPLE PROGRAMS
#    These reproduce the exact example from D1 §"Language Overview" and add a
#    second snippet to exercise the bracketed eval_target form.
#
#    Key grammar points reflected here (all from D1 EBNF):
#
#    (a) model declaration — opening brace on its OWN LINE:
#          model KNN knn_small
#          {
#              k = 3,
#              distance = "euclidean"
#          }
#        (Brace placement is a parser / formatting concern; the lexer just
#        emits LBRACE regardless of which line it appears on.)
#
#    (b) analyze overfitting — fields separated by NOTHING (no comma),
#        each field terminated by its own ";":
#          analyze overfitting
#          {
#              threshold = 0.10;
#              underfit_max_acc = 0.70;
#          }
#        EBNF: <analyze_field_list> ::= <analyze_field> { <analyze_field> }
#        There is NO "," between fields.  The ";" after the value belongs to
#        the field rule itself, not to any inter-field separator.
#
#    (c) top-level eval_stmt uses the DOTTED form of eval_target:
#          evaluate iris_exp.best on iris.test;
#        EBNF: <eval_target> ::= <identifier>
#                              | <identifier> "." <identifier>   ← used here
#                              | "[" <identifier> "." <identifier> "]"
# ─────────────────────────────────────────────────────────────────────────────

# ── Sample 1: exact D1 spec example ──────────────────────────────────────────
SAMPLE_PROGRAM = """\
// Dataset
dataset iris = load("iris.csv");

split iris into train(0.70), validation(0.15), test(0.15);

// Models
model KNN knn_small
{
    k = 3,
    distance = "euclidean"
}

model DecisionTree dt_main
{
    max_depth = 5,
    criterion = "gini"
}

// Experiment
experiment iris_exp
{
    train knn_small on iris.train;
    train dt_main on iris.train;

    evaluate knn_small on iris.validation;
    evaluate dt_main on iris.validation;

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

// Final evaluation and report
evaluate iris_exp.best on iris.test;

report summary
{
    metrics = [accuracy, f1],
    show = overfitting_analysis
}
"""

# ── Sample 2: bracketed eval_target form ─────────────────────────────────────
# Exercises the third eval_target alternative:
#   <eval_target> ::= "[" <identifier> "." <identifier> "]"
SAMPLE_BRACKET_EVAL = """\
// Bracketed eval_target form
evaluate [iris_exp.best] on iris.test;
"""


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN — pretty-print tokens for both samples
# ─────────────────────────────────────────────────────────────────────────────

def _print_tokens(label: str, source: str) -> None:
    """Lex `source`, then print every token in an aligned table."""
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")

    lexer = Lexer(source)
    try:
        tokens = lexer.tokenize()
    except LexicalError as err:
        print(err)
        return

    w_type  = max(len(t.type)  for t in tokens) + 2
    w_value = max(len(t.value) for t in tokens) + 2

    header = (
        f"{'TYPE':<{w_type}} "
        f"{'VALUE':<{w_value}} "
        f"{'LINE':>5}  "
        f"{'COL':>5}"
    )
    print(header)
    print("-" * len(header))

    for tok in tokens:
        print(
            f"{tok.type:<{w_type}} "
            f"{tok.value!r:<{w_value}} "
            f"{tok.line_number:>5}  "
            f"{tok.column_number:>5}"
        )

    print(f"\n  {len(tokens)} tokens produced.")


def main() -> None:
    _print_tokens("Sample 1 — D1 spec program (dotted eval_target)", SAMPLE_PROGRAM)
    _print_tokens("Sample 2 — bracketed eval_target form", SAMPLE_BRACKET_EVAL)


if __name__ == "__main__":
    main()