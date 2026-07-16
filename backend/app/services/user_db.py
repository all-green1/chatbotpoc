from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_USER_DB_PATH = "/data/user_db/users.json"


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    data: Dict[str, Any]


class UserDBService:
    """
    POC UserDB backed by a JSON file:

    File shape:
    {
      "user_123": {"grade": "A", "course": "CS", "previous_scores": [..], ...},
      "user_456": {...}
    }
    """

    def __init__(self, *, path: str = DEFAULT_USER_DB_PATH) -> None:
        self.path = path

    def get_user_profile(self, *, user_id: str) -> Optional[UserProfile]:
        user_id = (user_id or "").strip()
        if not user_id:
            return None

        if not os.path.exists(self.path):
            return None

        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            return None

        data = raw.get(user_id)
        if not isinstance(data, dict):
            return None

        return UserProfile(user_id=user_id, data=data)


def get_user_db_service() -> UserDBService:
    path = os.getenv("USER_DB_PATH", DEFAULT_USER_DB_PATH)
    return UserDBService(path=path)