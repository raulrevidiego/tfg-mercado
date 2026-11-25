"""Tono con FinBERT y tono de la cita del CEO.

FinBERT es un modelo de lenguaje (BERT) reentrenado con textos financieros: lee cada frase entera y da la
probabilidad de que sea positiva, negativa o neutra. Usa el contexto de la frase, no solo palabras sueltas como
el lexicón de `tone.py`, pero no es infalible: en la comprobación de `python -m backend.events.tone_finbert`
puntúa como negativa "Operating costs fell sharply while margins reached a record high" (que es buena noticia).

Dos decisiones, ambas configurables en `evaluate_tone`:
- método: "lexicon" (el baseline de `tone.py`) o "finbert" (modelo `ProsusAI/finbert`, gratuito; la primera vez
  se descarga, unos 440 MB, y después queda en la caché de Hugging Face).
- parte del comunicado: "all" (todas las frases, sin el aviso legal ni las tablas de cifras) o "quotes" (solo
  lo que se pone entre comillas, que en un comunicado de resultados son las palabras del CEO: justo el
  "comentario del CEO" que estudia el TFG).

El tono de un comunicado con FinBERT es la media de (P(positiva) − P(negativa)) de sus frases, de −1 a +1,
igual que el del lexicón; después se compara con los comunicados anteriores de la empresa (tono relativo).

`python -m backend.events.tone_finbert` puntúa unas frases de ejemplo para comprobar que el modelo funciona.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable

from backend.events.tone import strip_boilerplate, tone_score

MODEL = "ProsusAI/finbert"
METHODS = ("lexicon", "finbert")
PARTS = ("all", "quotes")

# Un "puntuador" recibe frases y devuelve (P positiva, P negativa, P neutra) de cada una.
Scorer = Callable[[list[str]], list[tuple[float, float, float]]]

_CURLY = re.compile(r"“([^”]{20,1500})”")
_STRAIGHT = re.compile(r'"([^"]{20,1500})"')


def extract_quotes(text: str, min_words: int = 8) -> list[str]:
    """Textos entre comillas (sin el aviso legal). Descarta los muy cortos ("record", "AI factory"...).

    Primero busca comillas tipográficas “así”; si no hay ninguna, comillas rectas "así". Con comillas rectas
    un número impar de ellas puede desajustar los pares: es una limitación conocida.
    """
    body = strip_boilerplate(text)
    found = _CURLY.findall(body) or _STRAIGHT.findall(body)
    quotes = [" ".join(q.split()) for q in found]
    return [q for q in quotes if len(q.split()) >= min_words]


def split_sentences(text: str, min_words: int = 6, max_chars: int = 600, max_sentences: int = 300) -> list[str]:
    """Frases del texto, sin los fragmentos cortos ni las filas de tablas de cifras (poca letra, mucho número)."""
    out = []
    for piece in re.split(r"(?<=[.!?])\s+|\n+", text):
        piece = " ".join(piece.split())
        if len(piece.split()) < min_words:
            continue
        if sum(c.isalpha() for c in piece) / len(piece) < 0.6:
            continue
        out.append(piece[:max_chars])
    return out[:max_sentences]


def finbert_tone(text: str, scorer: Scorer, part: str = "all") -> dict | None:
    """Tono de un comunicado con un puntuador FinBERT. None si no hay frases que puntuar."""
    if part == "quotes":
        sentences = [s for q in extract_quotes(text) for s in split_sentences(q, min_words=4)]
    else:
        sentences = split_sentences(strip_boilerplate(text))
    if not sentences:
        return None
    probs = scorer(sentences)
    n = len(probs)
    pos, neg, neu = (sum(p[i] for p in probs) / n for i in range(3))
    return {"pos": pos, "neg": neg, "neu": neu, "n_sentences": n, "score": pos - neg}


def load_scorer(model_name: str = MODEL, device: str = "auto", batch_size: int = 32, max_tokens: int = 128) -> Scorer:
    """Carga FinBERT (necesita `transformers` y PyTorch) y devuelve el puntuador. Usa la GPU si hay."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Falta transformers para FinBERT: pip install -e ".[nlp]" (PyTorch ya lo tienes)') from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    labels = {int(i): str(name).lower() for i, name in model.config.id2label.items()}
    try:  # el orden de las clases depende del modelo: se busca por nombre, no por posición
        order = [next(i for i, name in labels.items() if name.startswith(key)) for key in ("pos", "neg", "neu")]
    except StopIteration:
        raise RuntimeError(f"El modelo {model_name} no tiene clases positive/negative/neutral: {labels}") from None

    @torch.no_grad()
    def score(sentences: list[str]) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for start in range(0, len(sentences), batch_size):
            batch = tokenizer(
                sentences[start : start + batch_size], padding=True, truncation=True,
                max_length=max_tokens, return_tensors="pt",
            ).to(device)
            probs = torch.softmax(model(**batch).logits.float(), dim=-1).cpu().tolist()
            out.extend((p[order[0]], p[order[1]], p[order[2]]) for p in probs)
        return out

    return score


def make_tone_fn(
    method: str = "lexicon", part: str = "all", model: str = MODEL, device: str = "auto"
) -> Callable[[str], dict | None]:
    """Función texto -> {"score": ..., ...} (o None si el texto no tiene la parte pedida)."""
    if method not in METHODS:
        raise ValueError(f"Método de tono desconocido: {method!r} (usa {', '.join(METHODS)})")
    if part not in PARTS:
        raise ValueError(f"Parte del comunicado desconocida: {part!r} (usa {', '.join(PARTS)})")
    if method == "lexicon":
        def lexicon(text: str) -> dict | None:
            if part == "quotes":
                quotes = extract_quotes(text)
                return tone_score(" ".join(quotes)) if quotes else None
            return tone_score(text)

        return lexicon
    scorer = load_scorer(model, device)
    return lambda text: finbert_tone(text, scorer, part)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Comprueba FinBERT puntuando unas frases de ejemplo.")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    examples = [
        "Revenue grew 22% year over year, well above our guidance, driven by strong demand.",
        "Operating costs fell sharply while margins reached a record high.",
        "We recorded a significant impairment charge and lowered our full-year outlook.",
        "The company will hold its annual shareholder meeting on June 5 in Houston.",
    ]
    print(f"\nCargando {args.model} (la primera vez se descarga)...")
    try:
        scorer = load_scorer(args.model, args.device)
    except RuntimeError as exc:
        raise SystemExit(f"\n{exc}") from None
    print(f"\n{'positiva':>9}{'negativa':>10}{'neutra':>8}  frase")
    for sentence, (pos, neg, neu) in zip(examples, scorer(examples)):
        print(f"{pos:>9.2f}{neg:>10.2f}{neu:>8.2f}  {sentence}")
    print(
        "\nEsperado: 1.ª positiva, 3.ª negativa y 4.ª neutra. La 2.ª es un caso difícil (bajan los costes y suben los\n"
        "márgenes: buena noticia); si sale negativa es un error del modelo con frases así, no un fallo de la instalación.\n"
        "Si las otras tres salen mal, sí habría un problema (por ejemplo, clases invertidas)."
    )


if __name__ == "__main__":
    main()
