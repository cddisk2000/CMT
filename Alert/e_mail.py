def send_email(cfg: dict, subject: str, body: str):
    import smtplib
    from email.message import EmailMessage

    if not cfg.get("email_to") or not cfg.get("smtp_host"):
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.get("email_from") or cfg.get("smtp_user")
    msg["To"] = cfg.get("email_to")
    msg.set_content(body)

    with smtplib.SMTP(cfg["smtp_host"], cfg.get("smtp_port", 587)) as smtp:
        smtp.starttls()
        if cfg.get("smtp_user"):
            smtp.login(cfg["smtp_user"], cfg.get("smtp_pass", ""))
        smtp.send_message(msg)
