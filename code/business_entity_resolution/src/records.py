from __future__ import annotations

from dataclasses import dataclass

from .normalize import (
    addr_tokens,
    digit_sig,
    house_number,
    locality_tokens,
    name_prefix,
    name_tokens,
    postal_code,
)


@dataclass(slots=True)
class Record:
    eid: str
    country: str
    name_set: frozenset
    addr_set: frozenset
    postal: str
    house: str
    loc_set: frozenset
    prefix2: str
    prefix1: str
    core: str
    digits: frozenset
    name_join: str
    addr_join: str


def build_record(eid: str, name: str, address: str, country: str) -> Record:
    country = (country or "").strip()
    nt = name_tokens(name)
    at = addr_tokens(address)
    loc = locality_tokens(address)
    return Record(
        eid=str(eid),
        country=country,
        name_set=frozenset(nt),
        addr_set=frozenset(at),
        postal=postal_code(address, country),
        house=house_number(address),
        loc_set=frozenset(loc),
        prefix2=name_prefix(nt, 2),
        prefix1=name_prefix(nt, 1),
        core=" ".join(nt),
        digits=frozenset(digit_sig(name, address)),
        name_join=" ".join(nt),
        addr_join=" ".join(at),
    )
