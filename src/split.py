from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from mapping import GAME_TO_GENRE, GENRES


RAW_DIR_DEFAULT = Path("data/raw")
SPLITS_DIR_DEFAULT = Path("data/splits")


def discover_games(raw_dir: Path) -> List[str]:
    """
    Discover game folder names under raw_dir/train and raw_dir/test.
    We treat both as sources of images; we do NOT use Kaggle's split.
    """
    games = set()

    for sub in ["train", "test"]:
        p = raw_dir / sub
        if not p.exists():
            continue
        for child in p.iterdir():
            if child.is_dir():
                games.add(child.name)

    return sorted(games)


def validate_mapping(games: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns (mapped_games, unmapped_games)
    """
    mapped, unmapped = [], []
    for g in games:
        if g in GAME_TO_GENRE:
            mapped.append(g)
        else:
            unmapped.append(g)
    return mapped, unmapped


def pick_test_games(
    games_by_genre: Dict[str, List[str]],
    seed: int,
) -> List[str]:
    """
    Picks at least one game per genre for test, if possible.
    For genres with only 1 game available (should not happen here), it still picks it.
    """
    rng = random.Random(seed)
    test_games: List[str] = []

    for genre in GENRES:
        candidates = list(games_by_genre.get(genre, []))
        if not candidates:
            raise ValueError(f"No games found for genre '{genre}'. Check mapping/dataset.")
        rng.shuffle(candidates)
        test_games.append(candidates[0])

    test_games = sorted(set(test_games))
    return test_games


def make_split(
    all_games: List[str],
    seed: int,
    val_fraction: float,
    test_games: List[str],
) -> Dict[str, List[str]]:
    """
    Create game-level split:
      - test_games: fixed list
      - train_games: remaining games
      - val_images_fraction: stored in config JSON (validation is by images, not games)
    """
    all_set = set(all_games)
    test_set = set(test_games)

    missing = test_set - all_set
    if missing:
        raise ValueError(f"Test games not found in dataset folders: {sorted(missing)}")

    train_games = sorted(list(all_set - test_set))

    return {
        "train_games": train_games,
        "test_games": sorted(list(test_set)),
        "val_image_fraction": [val_fraction],  
        "seed": [seed],
        "genres": GENRES,
        "game_to_genre": GAME_TO_GENRE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create game-level split JSON files.")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR_DEFAULT))
    parser.add_argument("--splits-dir", type=str, default=str(SPLITS_DIR_DEFAULT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="split_v1.json")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    games = discover_games(raw_dir)
    if not games:
        raise SystemExit(
            f"No game folders found under {raw_dir}/train or {raw_dir}/test. "
            "Check dataset location."
        )

    mapped, unmapped = validate_mapping(games)
    if unmapped:
        raise SystemExit(
            "Some game folders are not in GAME_TO_GENRE mapping:\n"
            f"  {unmapped}\n"
            "Add them to src/mapping.py (GAME_TO_GENRE) first."
        )

    games_by_genre: Dict[str, List[str]] = defaultdict(list)
    for g in mapped:
        games_by_genre[GAME_TO_GENRE[g]].append(g)

    test_games = pick_test_games(games_by_genre, seed=args.seed)
    split = make_split(
        all_games=mapped,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_games=test_games,
    )

    out_path = splits_dir / args.out
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote split file: {out_path}")
    print(f"Test games ({len(test_games)}): {test_games}")
    print(f"Train games ({len(split['train_games'])}): {split['train_games'][:5]} ...")


if __name__ == "__main__":
    main()
