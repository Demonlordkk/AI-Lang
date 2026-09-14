"""Opcode definitions for the AI-Lang virtual machine.

Opcodes are small integers rather than strings so the interpreter loop can
dispatch through a jump table instead of a chain of string comparisons.

Specialised arithmetic opcodes (ADD_II, LT_II, ...) exist because the
compiler can often prove both operands are numbers. Those forms skip all
runtime type dispatch, which is the single largest cost in a hot loop.
"""

from __future__ import annotations

_NAMES = [
    # stack / constants
    "PUSH", "POP", "DUP",
    # names
    "LOAD", "LOAD_FAST", "STORE", "SET", "SET_FAST",
    # arithmetic (generic)
    "BINARY", "UNARY", "TO_BOOL",
    # arithmetic (specialised numeric fast paths)
    "ADD_NN", "SUB_NN", "MUL_NN", "DIV_NN", "MOD_NN",
    "LT_NN", "LE_NN", "GT_NN", "GE_NN",
    "EQ", "NE",
    "ADD_CONST", "INC_FAST",
    # data
    "MAKE_LIST", "MAKE_MAP", "INDEX", "FIELD", "SET_INDEX", "SET_FIELD",
    "CONVERT",
    # control flow
    "JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE",
    "JUMP_IF_FALSE_KEEP", "JUMP_IF_TRUE_KEEP", "JUMP_IF_SOME",
    # functions
    "CALL", "CALL_KW", "CLOSURE", "RETURN", "RETURN_NONE",
    # iteration
    "ITER_INIT", "ITER_NEXT", "ITER_BREAK", "ITER_END",
    "RANGE_INIT", "RANGE_NEXT",
    # scopes
    "SCOPE_PUSH", "SCOPE_POP",
    # errors
    "TRY_PUSH", "TRY_POP", "RAISE",
    # misc
    "PRINT", "RECORD", "IMPORT", "HALT",
    # superinstructions: fused pairs that remove stack round-trips
    "LOAD_LOAD",      # push two locals at once
    "LOAD_PUSH",      # push a local then a constant
    "LOAD_ADD_NN",    # stack value + local
    "LOAD_LT_NN",     # stack value < local
    "LOAD_FIELD",     # local.field in one step
    "LOAD_INDEX",     # index a stacked object by a local
]

# name -> int, and the reverse for disassembly / serialisation
OPS = {name: i for i, name in enumerate(_NAMES)}
NAMES = {i: name for name, i in OPS.items()}
globals().update(OPS)

__all__ = ["OPS", "NAMES"] + _NAMES
