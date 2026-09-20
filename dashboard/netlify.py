"""Netlify publishing: create a site once, then zip-deploy the dashboard bundle to it.

Only three REST calls are needed (create site, deploy zip, poll the deploy) plus delete for
"Unpublish". The token comes from settings.netlify_token and is only ever sent as a header;
it is never logged or written anywhere. Mirrors OllamaLLM.chat: requests + one typed error.
"""
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

API = "https://api.netlify.com/api/v1"
READY, FAILED = "ready", "error"


class PublishError(RuntimeError):
    """Raised when Netlify cannot be reached, rejects the request, or the deploy fails."""


@dataclass
class NetlifyClient:
    token: str
    timeout_s: int = 60

    def _headers(self, **extra: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "User-Agent": "local-data-agent", **extra}

    def _call(self, method, url: str, expect: tuple[int, ...] = (200, 201), **kw) -> requests.Response:
        try:
            r = method(url, headers=self._headers(**kw.pop("headers", {})), timeout=self.timeout_s, **kw)
        except requests.RequestException as exc:
            raise PublishError(f"Netlify request failed: {exc}") from exc
        if r.status_code not in expect:
            raise PublishError(f"Netlify {r.status_code}: {r.text[:200]}")
        return r

    def create_site(self, name: str) -> dict:
        return self._call(requests.post, f"{API}/sites", json={"name": name}).json()

    def deploy_zip(self, site_id: str, zip_bytes: bytes) -> dict:
        return self._call(requests.post, f"{API}/sites/{site_id}/deploys", data=zip_bytes,
                          headers={"Content-Type": "application/zip"}).json()

    def get_deploy(self, deploy_id: str) -> dict:
        return self._call(requests.get, f"{API}/deploys/{deploy_id}").json()

    def wait_ready(self, deploy_id: str, timeout_s: int = 120, poll_s: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            deploy = self.get_deploy(deploy_id)
            state = deploy.get("state")
            if state == READY:
                return deploy
            if state == FAILED:
                raise PublishError(deploy.get("error_message") or "Netlify deploy failed")
            if time.monotonic() >= deadline:
                raise PublishError(f"Netlify deploy did not become ready in {timeout_s} s (state: {state})")
            time.sleep(poll_s)

    def delete_site(self, site_id: str) -> None:
        self._call(requests.delete, f"{API}/sites/{site_id}", expect=(200, 204))


def site_name(slug: str) -> str:
    """Netlify subdomains are global: add a short random suffix to the dashboard slug."""
    return f"{slug[:40].strip('-') or 'dashboard'}-{secrets.token_hex(3)}"


def publish(client: NetlifyClient, zip_bytes: bytes, site_id: str | None, slug: str) -> dict:
    """First publish creates the site; later ones redeploy to the same site_id.
    -> {"site_id", "url", "deploy_id"}"""
    url = None
    if not site_id:
        site = client.create_site(site_name(slug))
        site_id, url = site.get("id"), site.get("ssl_url") or site.get("url")
        if not site_id:
            raise PublishError("Netlify did not return a site id")
    deploy = client.deploy_zip(site_id, zip_bytes)
    deploy_id = deploy.get("id")
    if not deploy_id:
        raise PublishError("Netlify did not return a deploy id")
    done = client.wait_ready(deploy_id)
    url = done.get("ssl_url") or done.get("url") or url
    if not url:
        raise PublishError("Netlify did not return the site URL")
    return {"site_id": site_id, "url": url, "deploy_id": deploy_id}
