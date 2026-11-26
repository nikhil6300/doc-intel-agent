"""Thin client for the Document Intelligence API, used by every eval script.
Accepts an existing httpx.Client so tests can pass FastAPI's TestClient."""

import time
from typing import Any

import httpx


class EvalClient:
    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    @classmethod
    def from_url(
        cls, url: str, api_key: str | None = None, timeout: float = 300.0, reviewer_key: str | None = None
    ) -> "EvalClient":
        headers = {"X-API-Key": api_key} if api_key else {}
        if reviewer_key:
            headers["X-Reviewer-Key"] = reviewer_key
        return cls(httpx.Client(base_url=url, headers=headers, timeout=timeout))

    def _call(self, method: str, path: str, **kwargs: Any) -> tuple[Any, float]:
        t0 = time.perf_counter()
        resp = self.http.request(method, path, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        return resp.json(), round(ms, 2)

    def search(self, question: str, k: int) -> tuple[list[dict], float]:
        data, ms = self._call("GET", "/v1/search", params={"q": question, "k": k})
        return data["results"], ms

    def query(self, question: str, top_k: int | None = None) -> tuple[dict, float]:
        body: dict = {"question": question}
        if top_k:
            body["top_k"] = top_k
        return self._call("POST", "/v1/query", json=body)

    def documents(self) -> list[dict]:
        return self._call("GET", "/v1/documents")[0]

    def document(self, doc_id: str) -> dict | None:
        try:
            return self._call("GET", f"/v1/documents/{doc_id}")[0]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def health(self) -> dict:
        return self._call("GET", "/health")[0]

    def review_export(self, statuses: str = "edited,rejected") -> list[dict]:
        return self._call("GET", "/v1/review/export", params={"statuses": statuses})[0]
