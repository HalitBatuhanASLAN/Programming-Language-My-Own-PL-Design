from ayar_lexer import Lexer, LexicalError, Token

# ── helper ────────────────────────────────────────────────────────────────────
def lex(source: str):
    """Lex a snippet and return the token list."""
    return Lexer(source).tokenize()

def types(source: str):
    """Return just the token types as a list — useful for quick assertions."""
    return [t.type for t in lex(source)]

def values(source: str):
    """Return just the token values as a list."""
    return [t.value for t in lex(source)]


# ── TEST 1: Keywords are not confused with identifiers ────────────────────────
# 'trainer' starts with 'train' but is NOT a keyword
toks = lex("train trainer training")
assert toks[0].type  == "KEYWORD",    "train must be KEYWORD"
assert toks[1].type  == "IDENTIFIER", "trainer must be IDENTIFIER"
assert toks[2].type  == "IDENTIFIER", "training must be IDENTIFIER"
print("TEST 1 PASSED – keywords vs identifiers")


# ── TEST 2: INT vs FLOAT distinction ──────────────────────────────────────────
toks = lex("42  3.14  0.5  .75  7.")
assert toks[0].type == "INT_LITERAL",   f"expected INT got {toks[0].type}"
assert toks[1].type == "FLOAT_LITERAL", f"expected FLOAT got {toks[1].type}"
assert toks[2].type == "FLOAT_LITERAL", f"expected FLOAT got {toks[2].type}"
assert toks[3].type == "FLOAT_LITERAL", f"expected FLOAT got {toks[3].type}"
assert toks[4].type == "FLOAT_LITERAL", f"expected FLOAT got {toks[4].type}"
print("TEST 2 PASSED – INT vs FLOAT literals")


# ── TEST 3: Bool literals ─────────────────────────────────────────────────────
toks = lex("true false truefalse")
assert toks[0].type == "BOOL_LITERAL", "true must be BOOL_LITERAL"
assert toks[1].type == "BOOL_LITERAL", "false must be BOOL_LITERAL"
assert toks[2].type == "IDENTIFIER",   "truefalse must be IDENTIFIER"
print("TEST 3 PASSED – bool literals")


# ── TEST 4: Metric literals ───────────────────────────────────────────────────
toks = lex("accuracy precision recall f1 f1_score")
assert toks[0].type == "METRIC_LITERAL", "accuracy → METRIC_LITERAL"
assert toks[1].type == "METRIC_LITERAL", "precision → METRIC_LITERAL"
assert toks[2].type == "METRIC_LITERAL", "recall → METRIC_LITERAL"
assert toks[3].type == "METRIC_LITERAL", "f1 → METRIC_LITERAL"
assert toks[4].type == "IDENTIFIER",     "f1_score → IDENTIFIER (not a metric)"
print("TEST 4 PASSED – metric literals")


# ── TEST 5: String literals ───────────────────────────────────────────────────
toks = lex('"hello world"  "with \\"escaped\\" quotes"')
assert toks[0].type  == "STRING_LITERAL"
assert toks[0].value == '"hello world"'
assert toks[1].type  == "STRING_LITERAL"
print("TEST 5 PASSED – string literals")


# ── TEST 6: Two-char operators are not split into two single-char tokens ──────
t = types("== != <= >=")
assert t == ["OP_EQ", "OP_NEQ", "OP_LTE", "OP_GTE"], f"got {t}"
print("TEST 6 PASSED – two-character operators")


# ── TEST 7: Comments are ignored, but line numbers still advance ──────────────
toks = lex("dataset // this is a comment\niris")
assert toks[0].value == "dataset" and toks[0].line_number == 1
assert toks[1].value == "iris"    and toks[1].line_number == 2
print("TEST 7 PASSED – comments ignored, line counter advances")


# ── TEST 8: Column numbers are accurate ──────────────────────────────────────
toks = lex("model KNN mymodel")
#           ^1    ^7  ^11
assert toks[0].column_number == 1,  f"'model' should start at col 1, got {toks[0].column_number}"
assert toks[1].column_number == 7,  f"'KNN' should start at col 7, got {toks[1].column_number}"
assert toks[2].column_number == 11, f"'mymodel' should start at col 11, got {toks[2].column_number}"
print("TEST 8 PASSED – column numbers")


# ── TEST 9: LexicalError is raised with correct location ─────────────────────
try:
    lex("dataset iris = @load")
    print("TEST 9 FAILED – should have raised LexicalError")
except LexicalError as e:
    assert e.char        == "@", f"wrong char: {e.char}"
    assert e.line_number == 1,   f"wrong line: {e.line_number}"
    assert e.col_number  == 16,  f"wrong col:  {e.col_number}"
    print(f"TEST 9 PASSED – LexicalError: {e}")


# ── TEST 10: Multi-line program with correct line tracking ────────────────────
source = (
    "dataset d = load(\"data.csv\");\n"   # line 1
    "// comment line\n"                   # line 2  (skipped)
    "split d into\n"                      # line 3
    "    train(0.7);\n"                   # line 4
)
toks = lex(source)
split_tok = next(t for t in toks if t.value == "split")
assert split_tok.line_number == 3, f"'split' should be on line 3, got {split_tok.line_number}"
print("TEST 10 PASSED – multi-line tracking across comments\n")


print("═" * 40)
print("ALL TESTS PASSED ✓")