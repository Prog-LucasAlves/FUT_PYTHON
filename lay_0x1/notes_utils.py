import uuid
from pathlib import Path

import pandas as pd
from app_config import NOTES_ASSETS_DIR, NOTES_FILE
from ui_constants import NOTE_PRIORITY_OPTIONS, NOTE_STATUS_OPTIONS

NOTE_COLUMNS = [
    "id",
    "created_at",
    "updated_at",
    "title",
    "note",
    "tag",
    "priority",
    "status",
    "pinned",
    "image_path",
]


def load_notes():
    if NOTES_FILE.exists():
        try:
            df = pd.read_json(NOTES_FILE, orient="records")
        except ValueError:
            return pd.DataFrame(columns=NOTE_COLUMNS)
    else:
        return pd.DataFrame(columns=NOTE_COLUMNS)

    for col in NOTE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["pinned"] = df["pinned"].fillna(False).astype(bool)
    df["image_path"] = df["image_path"].fillna("").astype(str)
    df["priority"] = df["priority"].fillna(NOTE_PRIORITY_OPTIONS[1])
    df["status"] = df["status"].fillna(NOTE_STATUS_OPTIONS[0])
    for col in ["tag", "title", "note", "created_at", "updated_at"]:
        df[col] = df[col].fillna("")
    return df[NOTE_COLUMNS].copy()


def save_notes(df_notes):
    df_notes = df_notes.copy()
    df_notes["pinned"] = df_notes["pinned"].fillna(False).astype(bool)
    if "image_path" not in df_notes.columns:
        df_notes["image_path"] = ""
    df_notes["image_path"] = df_notes["image_path"].fillna("").astype(str)
    df_notes.to_json(NOTES_FILE, orient="records", force_ascii=False, indent=2)


def save_note_attachment(uploaded_file, note_id):
    if uploaded_file is None:
        return ""
    NOTES_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".png"
    dest_path = NOTES_ASSETS_DIR / f"{note_id}{suffix}"
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest_path)


def new_note_id():
    return str(uuid.uuid4())
