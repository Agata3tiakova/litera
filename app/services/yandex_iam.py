import json
import time
import jwt
import requests
from threading import Lock


class YandexIAMTokenProvider:
    def __init__(self, sa_key_path: str):
        with open(sa_key_path, "r", encoding="utf-8") as f:
            self.sa_key = json.load(f)

        self._iam_token = None
        self._expires_at = 0
        self._lock = Lock()

    def get_token(self) -> str:
        with self._lock:
            if self._iam_token and time.time() < self._expires_at - 60:
                return self._iam_token

            self._refresh_token()
            return self._iam_token

    def _refresh_token(self):
        now = int(time.time())

        payload = {
            "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            "iss": self.sa_key["service_account_id"],
            "iat": now,
            "exp": now + 360
        }

        encoded_jwt = jwt.encode(
            payload,
            self.sa_key["private_key"],
            algorithm="PS256",
            headers={"kid": self.sa_key["id"]}
        )

        r = requests.post(
            "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            json={"jwt": encoded_jwt},
            timeout=10
        )
        r.raise_for_status()

        data = r.json()
        self._iam_token = data["iamToken"]
        self._expires_at = now + 11 * 60 * 60  # ~11 часов
