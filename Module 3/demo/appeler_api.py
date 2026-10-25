"""Appelle l'API locale avec les deux exemples documentés de validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx


DEMO_DIR = Path(__file__).resolve().parent
EXAMPLES_PATH = DEMO_DIR.parent / "app" / "exemples.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "url",
        nargs="?",
        default="http://127.0.0.1:8000",
        help="URL racine de l'API.",
    )
    return parser.parse_args()


def _print_prediction(label: str, example: dict[str, Any], result: dict[str, Any]) -> None:
    """Affiche une décision avec son score et la vérité terrain de l'exemple."""
    decision = "ALERTE" if result["alerte"] else "OK"
    print(
        f"{label} — score={result['probabilite']:.6f}, "
        f"seuil={result['seuil']:.6f}, décision={decision}, "
        f"vérité terrain={example['verite_terrain']}"
    )


def main() -> int:
    args = _parse_args()
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=10.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print("État de l'API :", json.dumps(health.json(), ensure_ascii=False))

        for name, label in (("fraude", "Exemple fraude"), ("legitime", "Exemple légitime")):
            example = examples[name]
            response = client.post(
                "/predict",
                json={"features": example["features"]},
            )
            response.raise_for_status()
            _print_prediction(label, example, response.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
