from __future__ import annotations

from typing import Dict, List

GENRES: List[str] = [
    "Shooter",
    "MOBA",
    "Sports",
    "RPG/MMORPG",
    "Sandbox/Survival",
]

GAME_TO_GENRE: Dict[str, str] = {
    "ApexLegends": "Shooter",
    "CSGO": "Shooter",
    "EscapeFromTarkov": "Shooter",
    "Overwatch": "Shooter",
    "PUBG_Battlegrounds": "Shooter",
    "Rainbows": "Shooter",          
    "Valorant": "Shooter",
    "Warzone": "Shooter",
    "Fortnite": "Shooter",
    "FreeFire": "Shooter",

    "Dota2": "MOBA",
    "LeagueOfLegends": "MOBA",
    "ClashRoyale": "MOBA",

    "FIFA21": "Sports",
    "RocketLeague": "Sports",

    "WorldOfWarcraft": "RPG/MMORPG",
    "GTAV": "RPG/MMORPG",
    "SeaOfThieves": "RPG/MMORPG",

    "Minecraft": "Sandbox/Survival",
    "Rust": "Sandbox/Survival",
    "DeathByDaylight": "Sandbox/Survival",
}

def normalize_game_name(game_folder: str) -> str:
    """
    Normalize game folder names if needed.
    Currently returns the same string, but kept for future-proofing.
    """
    return game_folder.strip()

def genre_for_game(game_folder: str) -> str:
    key = normalize_game_name(game_folder)
    if key not in GAME_TO_GENRE:
        raise KeyError(
            f"Unknown game folder '{game_folder}'. "
            f"Add it to GAME_TO_GENRE in src/mapping.py"
        )
    return GAME_TO_GENRE[key]
