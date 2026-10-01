"""Inspected COCO x86_64 little-endian raw ABI; actual bytes stay unmodified.

MsgHeader: seq@0, receive@8, type@16, sizeof24.
Book5: E@0, bids@8, asks@48, bidQty@88, askQty@108, status@128,
symbol@132, depths@148/149, sizeof152.
Trade: E@0, price@8, turnover@16, qty@24, total@28, status@32,
symbol@36, side@52, sizeof56. Padding is opaque, not data.
"""

from __future__ import annotations

import struct
from pathlib import Path

import zstandard

from .support import Event

HEADER = struct.Struct("<qqc7x")
BOOK = struct.Struct("<q5d5d5i5ii16sBB2x")
TRADE = struct.Struct("<qddiii16sB3x")


def _read(stream, size, *, eof=False):
    value = stream.read(size)
    while 0 < len(value) < size:
        chunk = stream.read(size - len(value))
        if not chunk:
            break
        value += chunk
    if not value and eof:
        return None
    if len(value) != size:
        raise ValueError("truncated native BIN header or payload")
    return value


def decode(stream, symbol: str):
    previous_receive, ordinal = 0, 0
    while (header := _read(stream, HEADER.size, eof=True)) is not None:
        sequence, raw_receive, kind = HEADER.unpack(header)
        if kind not in (b"B", b"T"):
            raise ValueError("unsupported native BIN message type")
        values = (BOOK if kind == b"B" else TRADE).unpack(_read(stream, BOOK.size if kind == b"B" else TRADE.size))
        encoded_symbol = values[22] if kind == b"B" else values[6]
        try:
            decoded_symbol = encoded_symbol.split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("invalid native symbol encoding") from error
        if decoded_symbol != symbol:
            raise ValueError("BIN payload symbol does not match fixed loader contract")
        receive = max(raw_receive, previous_receive)
        if receive <= 0:
            raise ValueError("nonpositive native receive clock")
        previous_receive = receive
        wire_exchange = values[0]
        exchange = wire_exchange if wire_exchange > 0 else receive
        common = {
            "kind": kind.decode("ascii"),
            "receive": receive,
            "exchange": exchange,
            "raw_receive": raw_receive,
            "wire_exchange": wire_exchange,
            "sequence": sequence,
            "ordinal": ordinal,
        }
        if kind == b"B":
            bid_depth, ask_depth = values[23:25]
            if bid_depth > 5 or ask_depth > 5:
                raise ValueError("BIN depth exceeds native Book5 capacity")
            yield Event(
                **common,
                status=values[21],
                bid=tuple(values[1:6][:bid_depth]),
                ask=tuple(values[6:11][:ask_depth]),
                bid_quantity=tuple(values[11:16][:bid_depth]),
                ask_quantity=tuple(values[16:21][:ask_depth]),
            )
        else:
            yield Event(**common, status=values[5], price=values[1], quantity=values[3], side=values[7])
        ordinal += 1


def events(path: Path, symbol: str):
    if not str(path).endswith(".bin.zst"):
        raise ValueError("prospective study requires the bound .bin.zst source; no fallback")
    with path.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
        yield from decode(stream, symbol)
