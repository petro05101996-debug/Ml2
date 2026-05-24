from pathlib import Path

FORBIDDEN = [
    "лучше соответствует",
    "балл соответствия",
    "вам подходит",
    "подходит вам",
    "соответствует вашим целям",
    "соответствует риск-профилю",
    "лучший вариант",
    "рекомендуем",
    "советуем",
    "покупайте",
    "продавайте",
    "держите",
    "с максимальным соответствием",
]

ALLOWED_FILES = {"investment_lab/engine/safety_text_guard.py"}


def test_static_wording_guard():
    roots = [Path("ui"), Path("backend"), Path("investment_lab")]
    bad = []
    for root in roots:
        for p in root.rglob("*"):
            if p.suffix not in {".py", ".ts", ".tsx", ".md"}:
                continue
            if p.as_posix() in ALLOWED_FILES:
                continue
            txt = p.read_text(encoding="utf-8")
            low = txt.casefold()
            for phrase in FORBIDDEN:
                if phrase in low:
                    bad.append(f"{p}: {phrase}")
    assert not bad, "\n".join(bad)
