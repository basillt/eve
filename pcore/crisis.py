"""
crisis.py — a deterministic safety net for self-harm / suicide messages.

A small offline model is warm but not reliable, especially in Arabic: it may listen kindly yet
forget to steer the person toward real help. Safety must not depend on the model guessing right.
So when a message clearly signals self-harm or suicide, the server guarantees a caring line that
affirms the person's life matters and turns them toward a trusted human or emergency help — in
their own language — appending it only when the model's own reply did not already do so.

This never provides methods and never blocks the conversation; it only adds the missing redirect.
"""

import re

# Clear signals of self-harm or suicidal intent. Kept specific, so ordinary sadness or venting
# ("I'm so tired", "تعبت") does not trigger it — only intent to die or to hurt oneself.
_EN = re.compile(
    r"\b(kill(ing)?\s+myself|end(ing)?\s+my\s+life|take\s+my\s+(own\s+)?life|"
    r"want\s+to\s+die|wish\s+i\s+(was|were)\s+dead|better\s+off\s+dead|"
    r"don'?t\s+want\s+to\s+(live|be\s+here|exist|wake\s+up)|no\s+reason\s+to\s+live|"
    r"hurt(ing)?\s+myself|harm(ing)?\s+myself|cut(ting)?\s+myself|suicid)\b",
    re.IGNORECASE,
)
_AR = [
    "اقتل نفسي", "أقتل نفسي", "اقتل حالي", "أقتل حالي", "انهي حياتي", "أنهي حياتي",
    "انهي حياتي", "اذي نفسي", "أذي نفسي", "اذي حالي", "أذي حالي", "اأذي حالي", "أأذي حالي",
    "اؤذي نفسي", "أؤذي نفسي", "اؤذي حالي", "بدي اموت", "بدي أموت", "بدي موت",
    "ما بدي اعيش", "ما بدي أعيش", "ما بدي اضل عايش", "مابدي اعيش", "انتحر", "أنتحر",
    "انتحار", "اخلص من حياتي", "أخلص من حياتي", "اجرح حالي", "أجرح حالي", "ما في فايدة من حياتي",
]

# A reply "steers" only when it points to a real human or to help — a trusted person, someone to
# talk to, emergency services. Pure comfort ("your life matters") is good but is NOT a redirect, so
# it is deliberately left out: a comfort-only reply still gets the net, which adds the redirect.
_STEER_EN = ("reach out", "reach someone", "someone you trust", "someone you can",
             "talk to someone", "talk to a", "not alone", "please reach", "emergency",
             "call someone", "call a", "please call", "hotline", "get help", "trusted",
             "help you find", "a professional", "a doctor", "a counselor", "a counsellor")
_STEER_AR = ("حدا", "شخص", "اتصل", "تتصل", "مساعدة", "طوارئ", "تثق", "قريب", "اطلب",
             "تواصل", "مختص", "طبيب", "دعم", "مش لحالك", "مش وحدك", "معك ناس")

_NOTE_AR = (
    "\n\nوأنا قلقانة عليك فعلاً. حياتك بتهمّ، وما بدّي تأذي حالك الليلة. في حدا بتثق فيه قريب منك "
    "تقدر تحكيله هلّأ؟ وإذا حسّيت إنك بخطر، تواصل مع خدمات الطوارئ عندك فوراً. أنا هون معك، "
    "وما رح أسيبك."
)
_NOTE_EN = (
    "\n\nI'm really worried about you. Your life matters, and I don't want you to hurt yourself "
    "tonight. Is there someone you trust nearby you could reach right now? And if you feel you're "
    "in danger, please contact your local emergency services straight away. I'm here with you."
)


def is_arabic(text):
    return sum("؀" <= ch <= "ۿ" for ch in text or "") >= 4


def is_crisis(text):
    t = text or ""
    if _EN.search(t):
        return True
    return any(p in t for p in _AR)


def already_steers(reply):
    r = (reply or "").lower()
    if any(w in r for w in _STEER_EN):
        return True
    return any(w in (reply or "") for w in _STEER_AR)


def safety_note(text):
    """The caring redirect to add, in the language the person wrote in."""
    return _NOTE_AR if is_arabic(text) else _NOTE_EN
