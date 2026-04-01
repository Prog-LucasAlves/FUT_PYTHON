from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

CACHE_VERSION = "2026-03-29-team-aliases-v11"

BETS_TRACKER_FILE = CURRENT_DIR / "bets_lay_tracker.csv"
NOTES_FILE = CURRENT_DIR / "notes.json"
NOTES_ASSETS_DIR = CURRENT_DIR / "notes_assets"
UNKNOWN_TEAMS_LOG_FILE = CURRENT_DIR / "unknown_team_names.log"
UNKNOWN_TEAMS_RESOLVED_FILE = CURRENT_DIR / "unknown_team_names_resolved.json"

DEFAULT_DATE = pd.Timestamp.today().normalize().date()
BET_STATUS_OPTIONS = ["", "Green", "75min"]
NOTE_PRIORITY_OPTIONS = ["Baixa", "Média", "Alta", "Urgente"]
NOTE_STATUS_OPTIONS = ["Aberta", "Em andamento", "Concluída", "Arquivada"]
