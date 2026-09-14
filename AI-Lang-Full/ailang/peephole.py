"""Bytecode peephole optimiser: fuse instruction pairs into superinstructions.

The VM spends a large share of its time pushing a value onto the operand
stack only to pop it again on the very next instruction. Fusing the common
pairs removes that round-trip entirely.

Safety rules, both of which must hold or the fusion is skipped:

1.  The second instruction of a pair must not be a jump target. Every jump in
    AI-Lang bytecode carries an absolute instruction index, so fusing two
    instructions would silently move the landing pad. We collect every target
    first and refuse to fuse into one.
2.  Indices shift when two instructions become one, so all jump operands are
    rewritten through a remap table afterwards.

The pass is purely local and preserves observable behaviour, including error
messages and the order in which operands are evaluated.
"""

from __future__ import annotations

from .opcodes import OPS

# Opcodes whose operand at the given tuple position is an absolute jump target.
_JUMP_TARGET_SLOT = {
    OPS["JUMP"]: 1,
    OPS["JUMP_IF_FALSE"]: 1,
    OPS["JUMP_IF_TRUE"]: 1,
    OPS["JUMP_IF_FALSE_KEEP"]: 1,
    OPS["JUMP_IF_TRUE_KEEP"]: 1,
    OPS["JUMP_IF_SOME"]: 1,
    OPS["ITER_NEXT"]: 1,
    OPS["ITER_BREAK"]: 1,
    OPS["RANGE_NEXT"]: 1,
    OPS["TRY_PUSH"]: 1,
}

_LOAD = OPS["LOAD"]
_PUSH = OPS["PUSH"]
_ADD_NN = OPS["ADD_NN"]
_LT_NN = OPS["LT_NN"]
_FIELD = OPS["FIELD"]
_INDEX = OPS["INDEX"]
_CALL = OPS["CALL"]

# (first, second) -> fused opcode. The fused instruction carries the operands
# of both halves, in order.
_PAIRS = {
    (_LOAD, _LOAD): OPS["LOAD_LOAD"],
    (_LOAD, _PUSH): OPS["LOAD_PUSH"],
    (_LOAD, _ADD_NN): OPS["LOAD_ADD_NN"],
    (_LOAD, _LT_NN): OPS["LOAD_LT_NN"],
    (_LOAD, _FIELD): OPS["LOAD_FIELD"],
    (_LOAD, _INDEX): OPS["LOAD_INDEX"],
}


def _jump_targets(code):
    """Every instruction index that some jump can land on."""
    targets = set()
    for ins in code:
        slot = _JUMP_TARGET_SLOT.get(ins[0])
        if slot is not None and len(ins) > slot:
            t = ins[slot]
            if isinstance(t, int):
                targets.add(t)
    return targets


def fuse(code):
    """Return a peephole-optimised copy of one instruction list."""
    if len(code) < 2:
        return list(code)

    targets = _jump_targets(code)

    out = []
    # old index -> new index, so jump operands can be rewritten afterwards
    remap = {}
    i = 0
    n = len(code)
    while i < n:
        remap[i] = len(out)
        a = code[i]
        if i + 1 < n and (i + 1) not in targets:
            b = code[i + 1]
            fused = _PAIRS.get((a[0], b[0]))
            if fused is not None:
                # The loaded local is the *right* operand of a fused binary
                # op: the left operand was pushed before the LOAD.
                out.append((fused,) + tuple(a[1:]) + tuple(b[1:]))
                remap[i + 1] = len(out) - 1
                i += 2
                continue
        out.append(a)
        i += 1
    remap[n] = len(out)

    # rewrite jump operands through the remap table
    for idx, ins in enumerate(out):
        slot = _JUMP_TARGET_SLOT.get(ins[0])
        if slot is not None and len(ins) > slot:
            t = ins[slot]
            if isinstance(t, int) and t in remap:
                lst = list(ins)
                lst[slot] = remap[t]
                out[idx] = tuple(lst)
    return out


def optimise_program(program):
    """Fuse `<main>` and every function body in place."""
    program.main.code = fuse(program.main.code)
    for fn in program.functions.values():
        fn.code = fuse(fn.code)
    return program
