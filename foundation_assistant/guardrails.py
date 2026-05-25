from dataclasses import dataclass


BLOCKED_PATTERNS = {
    "credential_theft": [
        "steal api key",
        "steal password",
        "phishing page",
        "bypass login",
    ],
    "malware": [
        "write ransomware",
        "keylogger",
        "reverse shell",
        "credential harvester",
    ],
    "violence": [
        "build a bomb",
        "make explosives",
        "assassinate",
    ],
    "privacy": [
        "doxx",
        "leak private",
        "find someone's address",
    ],
}

REDACTION_PATTERNS = {
    "api_key": ["sk-", "hf_"],
    "password": ["password=", "passwd=", "pwd="],
}


@dataclass
class GuardrailResult:
    allowed: bool
    category: str | None = None
    message: str | None = None


def check_user_input(text: str) -> GuardrailResult:
    lowered = text.lower()
    for category, patterns in BLOCKED_PATTERNS.items():
        for pattern in patterns:
            if pattern in lowered:
                return GuardrailResult(
                    allowed=False,
                    category=category,
                    message=(
                        "I cannot help with that request. I can still help with "
                        "safe alternatives, defensive guidance, or a high-level "
                        "explanation."
                    ),
                )
    return GuardrailResult(allowed=True)


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for label, patterns in REDACTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in redacted:
                redacted = redacted.replace(pattern, f"[REDACTED_{label.upper()}]")
    return redacted
