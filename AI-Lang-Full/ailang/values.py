"""Runtime value model and canonical display formatting for AI-Lang."""

from __future__ import annotations


class RecordType:
    """A record declaration; calling it constructs a RecordValue."""

    __slots__ = ("name", "fields", "field_types")

    def __init__(self, name, fields):
        self.name = name
        self.fields = [f for f, _ in fields]
        self.field_types = dict(fields)

    def __call__(self, *args, **kwargs):
        from .errors import VMError

        if len(args) > len(self.fields):
            raise VMError(f"{self.name} takes {len(self.fields)} fields, got {len(args)}")
        data = dict(zip(self.fields, args))
        for k, v in kwargs.items():
            if k not in self.fields:
                raise VMError(f"{self.name} has no field '{k}'")
            if k in data:
                raise VMError(f"duplicate value for field '{k}'")
            data[k] = v
        missing = [f for f in self.fields if f not in data]
        if missing:
            raise VMError(f"{self.name} missing field(s): {', '.join(missing)}")
        return RecordValue(self, data)

    def __repr__(self):
        return f"<record {self.name}>"


class RecordValue:
    __slots__ = ("rtype", "data")

    def __init__(self, rtype, data):
        self.rtype = rtype
        self.data = data

    @property
    def type_name(self):
        return self.rtype.name

    def get(self, name):
        from .errors import VMError

        if name not in self.data:
            raise VMError(f"{self.rtype.name} has no field '{name}'")
        return self.data[name]

    def set(self, name, value):
        from .errors import VMError

        if name not in self.data:
            raise VMError(f"{self.rtype.name} has no field '{name}'")
        self.data[name] = value

    def __eq__(self, other):
        return (
            isinstance(other, RecordValue)
            and other.rtype.name == self.rtype.name
            and other.data == self.data
        )

    def __hash__(self):
        return hash((self.rtype.name, tuple(sorted(self.data.items(), key=lambda kv: kv[0]))))

    def __repr__(self):
        inner = ", ".join(f"{k}: {display(v)}" for k, v in self.data.items())
        return f"{self.rtype.name}({inner})"


class Module:
    """A loaded AI-Lang module; members are accessed with dot notation."""

    __slots__ = ("name", "namespace")

    def __init__(self, name, namespace):
        self.name = name
        self.namespace = namespace

    def get(self, key):
        from .errors import VMError

        if key not in self.namespace:
            raise VMError(f"module '{self.name}' has no member '{key}'")
        return self.namespace[key]

    def __repr__(self):
        return f"<module {self.name}>"


def display(v) -> str:
    """Canonical text form used by `emit` and `str()`."""
    if v is None:
        return "nothing"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v == float("inf"):
            return "infinity"
        if v == float("-inf"):
            return "-infinity"
        if v == int(v) and abs(v) < 1e16:
            return f"{int(v)}.0"
        return repr(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "[" + ", ".join(_nested(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_nested(k)}: {_nested(x)}" for k, x in v.items()) + "}"
    if isinstance(v, RecordValue):
        return repr(v)
    if callable(v):
        name = getattr(v, "ailang_name", None) or getattr(v, "__name__", "function")
        return f"<function {name}>"
    return str(v)


def _nested(v) -> str:
    """Inside containers, text is quoted so output is unambiguous."""
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return display(v)


def type_name(v) -> str:
    if v is None:
        return "Nothing"
    if isinstance(v, bool):
        return "Bool"
    if isinstance(v, int):
        return "Int"
    if isinstance(v, float):
        return "Real"
    if isinstance(v, str):
        return "Text"
    if isinstance(v, list):
        return "List"
    if isinstance(v, dict):
        return "Map"
    if isinstance(v, RecordValue):
        return v.rtype.name
    if isinstance(v, RecordType):
        return "RecordType"
    if isinstance(v, Module):
        return "Module"
    if callable(v):
        return "Function"
    return type(v).__name__


def is_truthy(v) -> bool:
    """AI-Lang truthiness: only `false` and `nothing` are falsy."""
    if v is None or v is False:
        return False
    return True
