"""Engineering prototype; retain legacy scalar and NEL emission byte contracts."""

from __future__ import annotations

import yaml


def needs_legacy(value):
    if type(value) not in (dict, list, tuple):
        return True
    pending, seen = [value], set()
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind is str:
            if "\x85" in item or (not item.isascii() and any(0xD800 <= ord(character) <= 0xDFFF or ord(character) > 0xFFFF for character in item)):
                return True
        elif kind in (dict, list, tuple):
            if id(item) in seen:
                continue
            seen.add(id(item))
            if kind is dict:
                pending.extend(item.keys())
                pending.extend(item.values())
            else:
                pending.extend(item)
        elif kind not in (type(None), bool, int, float):
            return True
    return False


def compatible_bytes(value):
    dumper = yaml.SafeDumper if needs_legacy(value) else getattr(yaml, "CSafeDumper", yaml.SafeDumper)
    return yaml.dump(value, Dumper=dumper, sort_keys=True, allow_unicode=True, width=160).encode("utf-8")
