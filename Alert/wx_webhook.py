import json
import requests
import time


def send_webhook(url: str, payload: dict):
    if not url:
        return {"ok": False, "error": "webhook_url_empty"}
    url = url.strip()
    try:
        if "qyapi.weixin.qq.com/cgi-bin/webhook/send" in url:
            content = payload.get("content")
            if not content:
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
                content = "[CMT Alert Message]\n"
                content += f"Time: {ts}\n"
                content += f"Name: {node}\n"
                content += f"IP: {ip}\n"
                content += f"Status: {status}\n"
                content += f"RTT: {rtt_text}  Loss: {loss_text}\n"
                if status != "recovered":
                    content += f"Down Count: {count_text}"
            body = {
                "msgtype": "text",
                "text": {
                    "content": content,
                    "mentioned_mobile_list": ["@all"],
                },
            }
            resp = requests.post(
                url,
                headers={"Content-Type": "text/plain"},
                data=json.dumps(body),
                timeout=5.0,
            )
        else:
            resp = requests.post(url, json=payload, timeout=5.0)
        return {"ok": True, "status_code": resp.status_code, "text": resp.text}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
