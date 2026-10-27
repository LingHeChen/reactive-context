"""Conservative public-query hints; no gold, future queries or task solving."""

import re

_ORDINALS = {
    word: i
    for i, word in enumerate(
        (
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ),
        1,
    )
}
_DAY = re.compile(
    r"\bday[ -]*(\d+)\b|\b(\d+)(?:st|nd|rd|th)?[ -]*day\b|\b("
    + "|".join(_ORDINALS)
    + r")[ -]*day\b",
    re.I,
)
_FIELD = re.compile(
    r"\b(breakfast|lunch|dinner|accommodation|transportation|attraction|current_city)\b", re.I
)


def dependencies(query, people):
    """Bind fields/days locally to named people; avoid a cross-product of all mentions.

    This is a best-effort hint, not an authoritative parser of arbitrary English.
    It never invents values or removes unknown dependencies.
    """
    people = tuple(sorted(set(people)))
    if not people:
        return ()
    names = {p.casefold(): p for p in people}
    pattern = re.compile(r"\b(" + "|".join(re.escape(p) for p in people) + r")\b", re.I)
    result = set()
    for clause in re.split(r"[\n;!?]+|\.\s+", query):
        mentions = list(pattern.finditer(clause))
        for i, person in enumerate(mentions):
            end = mentions[i + 1].start() if i + 1 < len(mentions) else len(clause)
            suffix = clause[person.end() : end]
            prefix = clause[mentions[i - 1].end() if i else 0 : person.start()]
            days = list(_DAY.finditer(suffix))
            fields = list(_FIELD.finditer(suffix))
            if not days:
                # "breakfast on the second day ... join Joseph"
                days = list(_DAY.finditer(prefix))[-1:]
                fields = list(_FIELD.finditer(prefix))[-1:]
            elif not fields:
                # "dinner of Zoe on the 5th day"
                fields = list(_FIELD.finditer(prefix))[-1:]
            if not fields:
                continue
            for day in days:
                field = min(fields, key=lambda f: abs(f.start() - day.start()))
                number = int(day[1] or day[2]) if day[1] or day[2] else _ORDINALS[day[3].lower()]
                result.add((names[person[0].casefold()], number, field[0].lower()))
    return tuple(sorted(result))
