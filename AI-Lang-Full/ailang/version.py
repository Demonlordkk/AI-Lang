"""Single source of truth for AI-Lang version metadata."""

LANGUAGE = "AI-Lang"
VERSION = "2.10.0"
# AILBC-4 removes executable host-Python source from artifacts.  Artifacts
# produced by older formats must not be accepted by a newer runtime because
# AILBC-3 could carry native-loop source that the VM executed with exec().
BYTECODE_FORMAT = "AILBC-4"

__all__ = ["LANGUAGE", "VERSION", "BYTECODE_FORMAT"]
