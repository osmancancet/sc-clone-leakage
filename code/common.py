"""Shared paths, Solidity tokenisation and normalisation helpers."""
import hashlib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
# Large intermediate files live on local disk; override with SC_WORK.
WORK = Path(os.environ.get("SC_WORK", "/home/user/scwork"))
PROC = WORK / "processed"
RESULTS = ROOT / "results"
for p in (PROC, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(10))

# Comments and string literals are matched together so that "//" inside a
# string is not treated as a comment.
_COMMENT_OR_STRING = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)
_TOKEN = re.compile(
    r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''   # string literal
    r'|0x[0-9a-fA-F]+|\d+(?:\.\d+)?(?:e\d+)?'        # number
    r'|[A-Za-z_$][A-Za-z0-9_$]*'                      # identifier / keyword
    r'|==|!=|<=|>=|&&|\|\||\+\+|--|\+=|-=|\*=|/=|=>|\*\*|<<|>>|[^\sA-Za-z0-9_$]')

# Solidity keywords, types and security-relevant built-ins are kept verbatim
# in the type-2 normalisation; every other identifier becomes ID.
KEEP = set("""
pragma solidity import contract interface library is abstract function modifier
event emit struct enum mapping public private internal external pure view payable
constant immutable override virtual returns return if else for while do break
continue throw require assert revert new delete memory storage calldata using
constructor fallback receive try catch assembly let unchecked indexed anonymous
address bool string bytes byte int uint var true false this super selfdestruct
suicide msg sender value data sig gas gasleft tx origin gasprice block timestamp
number coinbase difficulty prevrandao gaslimit blockhash basefee chainid now
call delegatecall staticcall callcode transfer send balance code codehash
keccak256 sha3 sha256 ripemd160 ecrecover addmod mulmod abi encode encodePacked
encodeWithSelector encodeWithSignature decode wei gwei szabo finney ether
seconds minutes hours days weeks years length push pop type
""".split())
_INT_TYPE = re.compile(r'^(u?int|bytes|fixed|ufixed)\d*(x\d+)?$')


def strip_comments(src: str) -> str:
    def repl(m):
        s = m.group(0)
        return s if s[0] in "\"'" else " "
    return _COMMENT_OR_STRING.sub(repl, src)


def tokens(src: str):
    return _TOKEN.findall(strip_comments(src))


def type2(toks):
    out = []
    for t in toks:
        c = t[0]
        if c in "\"'" or c.isdigit():
            out.append("LIT")
        elif c.isalpha() or c in "_$":
            out.append(t if (t in KEEP or _INT_TYPE.match(t)) else "ID")
        else:
            out.append(t)
    return out


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_PRAGMA = re.compile(r'pragma\s+solidity\s*([^;]+);')
_VER = re.compile(r'0\.(\d+)\.(\d+)')


def solidity_minor(src: str):
    """Lowest 0.x minor version named in the first pragma, or None."""
    m = _PRAGMA.search(src)
    if not m:
        return None
    v = _VER.findall(m.group(1))
    if not v:
        return None
    return min(int(a) for a, _ in v)
