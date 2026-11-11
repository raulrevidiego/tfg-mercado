"""Tono de un comunicado: ¿suena positivo o negativo?

Versión 1, deliberadamente sencilla y transparente: se cuentan palabras de un pequeño lexicón
financiero propio (con la negación más simple: "not strong" cuenta como negativa). Sirve de BASELINE;
un modelo de lenguaje como FinBERT, que entiende el contexto, tendrá que mejorarlo para justificarse.

Dos precauciones:
- Se recorta el aviso legal ("forward-looking statements"), que habla de riesgos en TODOS los comunicados.
- El tono absoluto de una empresa suele ser siempre optimista, así que lo informativo es el **tono
  relativo**: cuánto se desvía un comunicado de los anteriores de la misma empresa.
"""

from __future__ import annotations

import re

POSITIVE = frozenset("""
record strong stronger strength growth grew grow growing increase increased increases increasing improve
improved improvement improvements exceeded exceed exceeds outperform outperformed robust momentum accelerate
accelerated acceleration raised raise raising expand expanded expansion gain gains gained profitable
profitability successful successfully success leading favorable benefit benefits benefited opportunity
opportunities optimistic confident confidence healthy solid surpassed highest best achieved achievement
""".split())

NEGATIVE = frozenset("""
decline declined declines declining decrease decreased decreases lower weak weaker weakness loss losses
impairment impairments restructuring headwind headwinds pressure pressures challenging challenges challenge
adverse adversely slowdown slowing slowed below missed miss shortfall deteriorate deteriorated deterioration
volatility reduced reduction reductions layoffs litigation investigation downturn softness soft disappointing
unfavorable cut cuts delay delayed delays inflation
""".split())

NEGATORS = frozenset({"not", "no", "never", "without", "neither", "nor"})
SAFE_HARBOR = re.compile(r"forward[- ]looking statements?", re.I)


def strip_boilerplate(text: str) -> str:
    """Quita el aviso legal de declaraciones prospectivas.

    Si aparece en el último 75 % del texto (lo habitual), se corta desde ahí hasta el final; si aparece
    al principio, se quitan los 2.500 caracteres siguientes.
    """
    m = SAFE_HARBOR.search(text)
    if m is None:
        return text
    if m.start() / max(len(text), 1) > 0.25:
        return text[: m.start()]
    return text[: m.start()] + text[m.start() + 2500 :]


def tone_score(text: str) -> dict:
    """Cuenta palabras positivas y negativas. `score` va de -1 (todo negativo) a +1 (todo positivo)."""
    words = re.findall(r"[a-z][a-z'-]*", strip_boilerplate(text).lower())
    pos = neg = 0
    for i, w in enumerate(words):
        sign = 1 if w in POSITIVE else -1 if w in NEGATIVE else 0
        if sign and any(x in NEGATORS for x in words[max(0, i - 3) : i]):
            sign = -sign
        pos += sign == 1
        neg += sign == -1
    total = pos + neg
    return {"pos": pos, "neg": neg, "n_words": len(words), "score": (pos - neg) / total if total else 0.0}


def relative_tone(scores: list[float], window: int = 8, min_prev: int = 3) -> list[float | None]:
    """Tono de cada comunicado menos la media de los `window` anteriores de la MISMA empresa.

    `scores` va en orden cronológico. Los primeros comunicados (con menos de `min_prev` anteriores)
    no tienen referencia y devuelven None.
    """
    out: list[float | None] = []
    for i, s in enumerate(scores):
        prev = scores[max(0, i - window) : i]
        out.append(s - sum(prev) / len(prev) if len(prev) >= min_prev else None)
    return out
