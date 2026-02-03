import time
from urllib.parse import parse_qs, urlparse

import requests


def _build_content(payload: dict) -> str:
    content = payload.get("content")
    if content:
        return content
    node = payload.get("name") or payload.get("node_id") or "node"
    status = payload.get("status", "unknown")
    ip = payload.get("ip", "")
    down_count = payload.get("down_count")
    site = payload.get("site") or "-"
    floor = payload.get("floor") or "-"
    line = payload.get("line") or "-"
    rtt_ms = payload.get("rtt_ms")
    loss = payload.get("loss")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    rtt_text = f"{rtt_ms:.1f}ms" if isinstance(rtt_ms, (int, float)) else "--"
    loss_text = f"{loss:.0f}%" if isinstance(loss, (int, float)) else "--"
    count_text = f"{down_count}" if down_count is not None else "--"
    lines = [
        "[CMT Alert Message]",
        f"Time: {ts}",
        f"Name: {node}",
        f"IP: {ip}",
        f"Status: {status}",
        f"RTT: {rtt_text}  Loss: {loss_text}",
        f"Site: {site}",
        f"Floor: {floor}",
        f"Line: {line}",
    ]
    if status != "recovered":
        lines.append(f"Down Count: {count_text}")
    return "\n".join(lines)


def _extract_params(url_or_token: str) -> dict:
    if not url_or_token:
        return {}
    if url_or_token.lower().startswith("http"):
        parsed = urlparse(url_or_token)
        query = parse_qs(parsed.query)
        params = {k: v[0] for k, v in query.items() if v}
        if "token" not in params and parsed.path:
            path_token = parsed.path.rstrip("/").split("/")[-1]
            if path_token and path_token.lower() != "send":
                params["token"] = path_token
        return params
    return {"token": url_or_token.strip()}


def send_pushplus_webhook(url_or_token: str, payload: dict):
    if not url_or_token:
        return {"ok": False, "error": "pushplus_token_empty"}
    params = _extract_params(url_or_token)
    token = params.get("token", "")
    if not token:
        return {"ok": False, "error": "pushplus_token_missing"}
    content = _build_content(payload)
    title = payload.get("title") or params.get("title") or "CMT Alert"
    template = payload.get("template") or params.get("template") or "html"
    topic = payload.get("topic") or params.get("topic")
    channel = payload.get("channel") or params.get("channel")
    webhook = payload.get("webhook") or params.get("webhook")
    callback_url = payload.get("callbackUrl") or params.get("callbackUrl")
    timestamp = payload.get("timestamp") or params.get("timestamp")
    pre = payload.get("pre") or params.get("pre")
    request_params = {"token": token, "content": content, "title": title, "template": template}
    if topic:
        request_params["topic"] = topic
    if channel:
        request_params["channel"] = channel
    if webhook:
        request_params["webhook"] = webhook
    if callback_url:
        request_params["callbackUrl"] = callback_url
    if timestamp:
        request_params["timestamp"] = timestamp
    if pre:
        request_params["pre"] = pre
    try:
        resp = requests.get(
            "https://www.pushplus.plus/send",
            params=request_params,
            timeout=5.0,
        )
        return {"ok": True, "status_code": resp.status_code, "text": resp.text}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
