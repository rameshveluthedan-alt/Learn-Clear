"""
LearnClear — AI Tutor for Indian School Students
=================================================
Built by : [Your Full Name]
Contact  : [Your Email]
Date     : March 2026

Tech Stack : Python, Google Gemini 2.5 Flash, pyTelegramBotAPI, Flask
Hosting    : Google Cloud Run (min-instances=1, no cold starts)
Telegram   : https://t.me/learn_clear_bot
Live App   : https://learn-clear-xxxxxxxx-el.a.run.app

Deployment:
  gcloud run deploy learn-clear \
    --source . \
    --region asia-south1 \
    --allow-unauthenticated \
    --min-instances 1 \
    --set-env-vars TELEGRAM_TOKEN=xxx,GEMINI_API_KEY=xxx
"""

import os
import re
import time
import logging
import threading

import telebot
from flask import Flask
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# 0.  BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("learn-clear")

BOT_TOKEN  = os.environ["TELEGRAM_TOKEN"]
GEMINI_KEY = os.environ["GEMINI_API_KEY"]

bot    = telebot.TeleBot(BOT_TOKEN, threaded=True)
client = genai.Client(api_key=GEMINI_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# 1a.  LANGUAGE STORE
# ──────────────────────────────────────────────────────────────────────────────
_user_lang: dict[int, str] = {}

SUPPORTED_LANGUAGES = {
    "1":  "English",
    "2":  "Hindi",
    "3":  "Bengali",
    "4":  "Tamil",
    "5":  "Telugu",
    "6":  "Marathi",
    "7":  "Gujarati",
    "8":  "Kannada",
    "9":  "Malayalam",
    "10": "Punjabi",
    "11": "Odia",
    "12": "Urdu",
}


def _get_lang(chat_id: int) -> str:
    return _user_lang.get(chat_id, "auto")


def _lang_instruction(chat_id: int, sample_text: str = "") -> str:
    lang = _get_lang(chat_id)
    if lang != "auto":
        return (
            f"LANGUAGE RULE: You MUST write your entire response in {lang}. "
            "Keep all HTML tags in ASCII but all visible text in that language. "
            "Do not switch languages mid-response."
        )
    if sample_text.strip():
        return (
            "LANGUAGE RULE: Detect the language of the student's message and reply "
            "entirely in that same language (Hindi, Tamil, Telugu, Bengali, etc.). "
            "Keep HTML tags in ASCII; all visible text must be in the detected language. "
            "Do not mix languages."
        )
    return "LANGUAGE RULE: Respond in English."


# ──────────────────────────────────────────────────────────────────────────────
# 1b.  CONVERSATION HISTORY STORE
# ──────────────────────────────────────────────────────────────────────────────
# Stores last Q&A per chat for follow-up context
# Format: { chat_id: { "question": str, "answer": str } }
_last_qa: dict[int, dict] = {}

MAX_HISTORY_CHARS = 1500   # trim long answers before putting in follow-up prompt


# ──────────────────────────────────────────────────────────────────────────────
# 1c.  SCOPE GUARD & PROMPTS
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED_SUBJECTS = {
    "mathematics", "maths", "math",
    "physics", "chemistry", "biology", "science",
    "history", "geography", "civics", "political science",
    "english", "hindi", "sanskrit",
    "economics", "accountancy", "business studies",
    "computer science", "information technology",
    "environmental science", "evs",
}

SCOPE_GUARD = """\
SCOPE RULE — STRICTLY ENFORCED:
You ONLY answer questions from these school subjects for Classes 1 to 12:
Mathematics, Physics, Chemistry, Biology, History, Geography, Civics,
English, Hindi, Sanskrit, Economics, Accountancy, Business Studies,
Computer Science, Environmental Science.

If the question is about ANYTHING else — current events, personal advice,
coding projects, general knowledge outside school syllabus, adult topics,
or any non-academic subject — respond ONLY with:

<i>I can only help with school subjects for Classes 1–12.
Please ask a subject-related question.</i>

Do not attempt to answer out-of-scope questions under any circumstances.
"""


def build_learnclear_image_prompt(lang_rule: str) -> str:
    """Prompt for photo/PDF input — textbook page, exam paper, worksheet."""
    return f"""\
You are LearnClear, a friendly and patient AI tutor for Indian school
students from Classes 1 to 12 (CBSE, ICSE, and State Board curricula).
{lang_rule}

{SCOPE_GUARD}

A student has sent you a photo or PDF of a question, problem, or topic
from their textbook or exam paper. Your job is to:

  1. Identify the subject and approximate class level from the content.
  2. Identify every question or problem visible in the image.
  3. Explain the underlying concept clearly before solving.
  4. Solve each problem step by step — never skip steps.
  5. State which formula, rule, or concept you are applying at each step.
  6. Give the final answer clearly with correct units where applicable.
  7. Add a memory tip or trick at the end to help the student remember.

════════════════════════════════════════════════════════
STRICT OUTPUT FORMAT — HTML ONLY
You MUST follow every rule below without exception.
════════════════════════════════════════════════════════

RULE 1 — CONFIDENCE LABEL (first line, always):
Choose ONE of these based on your certainty:
✅ <i><b>Standard NCERT/textbook concept — high confidence.</b></i>
⚠️ <i><b>Please verify this answer with your textbook or teacher before using in an exam.</b></i>

RULE 2 — SUBJECT & CLASS IDENTIFICATION:
<u><b>Subject: [Subject Name] | Class: [Estimated Class Level]</b></u>

RULE 3 — CONCEPT EXPLANATION (always before the solution):
<u><b>Concept</b></u>
<b>Topic</b>: <code>Name of the concept or chapter</code>
Explain the concept in 2–3 simple sentences in plain language.

RULE 4 — STEP BY STEP SOLUTION:
<u><b>Solution</b></u>
<b>Step 1</b>: [What you are doing and why]
<b>Step 2</b>: [Next step with working shown]
— and so on until the final answer.

RULE 5 — FORMULA DISPLAY:
Every formula must be shown explicitly before substitution.
Example: <b>Formula</b>: <code>Speed = Distance ÷ Time</code>

RULE 6 — NUMERICAL VALUES:
Wrap every number and unit inside <code> tags.
Example: Distance = <code>120 km</code>, Time = <code>2 hours</code>,
Speed = <code>60 km/h</code>

RULE 7 — FINAL ANSWER:
<u><b>Answer</b></u>
State the answer clearly and prominently.
For MCQs: state the correct option and explain why the other options are wrong.

RULE 8 — MEMORY TIP:
<u><b>💡 Remember</b></u>
One simple tip, trick, mnemonic, or shortcut to help the student
remember this concept for the exam.

RULE 9 — DISCLAIMER (last line, always — translate into response language):
<i>LearnClear is an AI tutor. Always verify important answers
with your textbook or teacher before an exam.</i>

RULE 10 — BANNED FORMATTING:
Do NOT use Markdown. No **, no __, no #, no *, no backtick outside <code>.

RULE 11 — NEVER DO THIS:
- Never give just the answer without showing working
- Never say "the answer is obvious" or skip explanation
- Never answer questions about guess papers, leaked papers, or cheating
- Never express opinions on teachers, schools, or exams
- Never diagnose learning disabilities or give personal advice
- If genuinely uncertain about any fact, say so clearly and tell
  the student to verify with their textbook

════════════════════════════════════════════════════════
SUBJECT-SPECIFIC INSTRUCTIONS:
════════════════════════════════════════════════════════

MATHEMATICS:
- Show every calculation step, no mental math shortcuts
- If multiple methods exist, show the standard textbook method first
- For geometry: describe the construction or diagram in words
- For word problems: first extract the given data, then solve

PHYSICS & CHEMISTRY:
- Always write units at every step, not just the final answer
- For chemistry equations: balance the equation before solving
- For physics: always state which law or principle applies
- Significant figures: match the precision given in the question

BIOLOGY:
- Base answers strictly on NCERT diagrams and definitions
- For diagrams asked in questions: describe what should be drawn
- For processes (photosynthesis, digestion etc.): use numbered steps

HISTORY & CIVICS & GEOGRAPHY:
- Base answers strictly on NCERT textbook content
- For dates and events: be precise, do not estimate
- If a fact is debated among historians, present the NCERT version
  and note that other perspectives exist
- For map-based questions: describe locations clearly in words

ENGLISH & HINDI & SANSKRIT:
- For grammar questions: state the rule first, then apply it
- For comprehension: quote the relevant line from the passage
- For essay/letter writing: provide structure and key points,
  not a complete written answer (student must write their own)
- For poetry: explain meaning, literary devices, and theme separately

ECONOMICS & ACCOUNTANCY:
- For numerical problems: show journal entries or calculations fully
- For theory: distinguish between short answer and long answer depth
- Use correct technical terminology from the NCERT syllabus
"""


def build_learnclear_text_prompt(lang_rule: str, student_question: str) -> str:
    """Prompt for typed text questions — doubt, concept, definition."""
    return f"""\
You are LearnClear, a friendly and patient AI tutor for Indian school
students from Classes 1 to 12 (CBSE, ICSE, and State Board curricula).
{lang_rule}

{SCOPE_GUARD}

A student has typed a question or doubt. Your job is to:
  1. Identify the subject and class level this question belongs to.
  2. Explain the concept clearly in plain language — no jargon.
  3. Use a simple real-life example the student can relate to.
  4. If it is a problem, solve it step by step.
  5. End with a memory tip to help the student retain the answer.

════════════════════════════════════════════════════════
STRICT OUTPUT FORMAT — HTML ONLY
════════════════════════════════════════════════════════

RULE 1 — CONFIDENCE LABEL (first line, always):
✅ <i><b>Standard NCERT/textbook concept — high confidence.</b></i>
OR
⚠️ <i><b>Please verify this answer with your textbook or teacher.</b></i>

RULE 2 — SUBJECT IDENTIFICATION:
<u><b>Subject: [Subject] | Topic: [Chapter/Topic Name]</b></u>

RULE 3 — SIMPLE EXPLANATION:
<u><b>What is [Term/Concept]?</b></u>
Explain in 3–4 sentences using simple language.
Avoid technical jargon — use words a Class 8 student understands.

RULE 4 — REAL LIFE EXAMPLE:
<u><b>🌍 Real Life Example</b></u>
Give one relatable everyday example that makes the concept click.
Prefer examples from Indian daily life where possible.

RULE 5 — EXAM ANGLE:
<u><b>📝 How This Appears in Exams</b></u>
Show one typical exam question on this topic and its model answer
in 1–2 lines — so the student knows what to expect.

RULE 6 — MEMORY TIP:
<u><b>💡 Remember</b></u>
One mnemonic, shortcut, or trick to remember this for the exam.

RULE 7 — DISCLAIMER (last line, always — translate into response language):
<i>LearnClear is an AI tutor. Always verify important answers
with your textbook or teacher before an exam.</i>

RULE 8 — BANNED FORMATTING:
Do NOT use Markdown. No **, no __, no #, no *, no backtick outside <code>.

RULE 9 — NEVER DO THIS:
- Never give a one-line answer without explanation
- Never answer questions unrelated to school subjects
- Never complete assignments, essays or projects for the student
  (guide them, don't write it for them)
- Never express uncertainty without flagging it with ⚠️
- For essay/creative writing: give structure and key points only —
  the student must write their own work

Student question: {student_question}
"""


def build_learnclear_followup_prompt(
        lang_rule: str,
        original_question: str,
        bot_answer: str,
        followup: str) -> str:
    """Prompt for follow-up questions in the same conversation."""
    return f"""\
You are LearnClear, a patient AI tutor for Indian school students.
{lang_rule}

{SCOPE_GUARD}

A student asked a follow-up question about your previous explanation.

ORIGINAL QUESTION:
{original_question}

YOUR PREVIOUS ANSWER (summary):
{bot_answer[:MAX_HISTORY_CHARS]}

STUDENT'S FOLLOW-UP:
{followup}

Your job:
1. Identify exactly what the student did not understand
2. Re-explain ONLY that specific part — do not repeat the full answer
3. Try a completely different approach — different words,
   different example, simpler breakdown
4. If the student says your answer was wrong,
   re-check carefully and either correct yourself or
   explain why your original answer was right

════════════════════════════════════════════════════════
OUTPUT FORMAT — HTML ONLY, same rules as before.
════════════════════════════════════════════════════════

Start with:
<u><b>Let me explain that differently —</b></u>

End with the disclaimer:
<i>LearnClear is an AI tutor. Always verify with your textbook or teacher.</i>

NEVER say "as I explained before" — re-explain freshly and simply.
NEVER use Markdown. No **, no __, no #, no *.
"""


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GEMINI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_GEMINI_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
)

_MODEL_PRIMARY  = "gemini-3.1-flash-lite-preview"
_MODEL_FALLBACK = "gemini-2.0-flash"


def _gemini_generate(parts: list) -> str:
    for attempt, model in enumerate((_MODEL_PRIMARY, _MODEL_FALLBACK), start=1):
        try:
            log.info("Gemini attempt %d — model: %s", attempt, model)
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=_GEMINI_CONFIG,
            )
            return response.text
        except Exception as exc:
            log.warning("Model %s failed (attempt %d): %s", model, attempt, exc)
            if attempt == 1:
                time.sleep(1.5)
    raise RuntimeError("Both Gemini models failed.")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  HTML SANITISER
# ──────────────────────────────────────────────────────────────────────────────

def _escape_text_nodes(text: str) -> str:
    parts = re.split(r"(</?[a-zA-Z][^>]*?>)", text)
    result = []
    for part in parts:
        if re.match(r"</?[a-zA-Z][^>]*?>", part):
            result.append(part)
        else:
            part = part.replace("&", "&amp;")
            part = part.replace("<", "&lt;")
            part = part.replace(">", "&gt;")
            result.append(part)
    return "".join(result)


def _remove_unsupported_tags(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    allowed = {"b", "i", "u", "s", "code", "pre", "a"}
    def _remove(m):
        tag = re.sub(r"[</> ]", "", m.group(0)).split()[0].lower().rstrip("/")
        return m.group(0) if tag in allowed else ""
    return re.sub(r"</?[a-zA-Z][^>]*?>", _remove, text)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`<>]+)`", r"<code>\1</code>", text)
    return text


def _ensure_disclaimer(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("✅") or stripped.startswith("⚠️"):
        return text
    if "✅" in stripped[:200] or "⚠️" in stripped[:200]:
        return text
    disclaimer = (
        "⚠️ <i><b>Please verify this answer with your textbook "
        "or teacher before an exam.</b></i>\n\n"
    )
    return disclaimer + text


def sanitize_for_telegram(text: str) -> str:
    text = _strip_markdown(text)
    text = _remove_unsupported_tags(text)
    text = _escape_text_nodes(text)
    text = _ensure_disclaimer(text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  HALLUCINATION DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

_UNCERTAINTY_PHRASES = [
    "i think", "i believe", "i'm not sure",
    "i am not sure", "probably", "i may be wrong",
    "not certain", "might be", "possibly",
]


def check_confidence(text: str) -> str:
    """
    If Gemini expresses uncertainty in its own words,
    append a verification reminder automatically.
    """
    if any(phrase in text.lower() for phrase in _UNCERTAINTY_PHRASES):
        text += (
            "\n\n⚠️ <i>This answer contains some uncertainty. "
            "Please verify with your textbook or teacher "
            "before using it in an exam.</i>"
        )
    return text


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SAFE REPLY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_MAX_MSG_LEN = 4096


def _send_long(chat_id: int, text: str, reply_to: int | None = None) -> None:
    chunks = [text[i: i + _MAX_MSG_LEN] for i in range(0, len(text), _MAX_MSG_LEN)]
    for idx, chunk in enumerate(chunks):
        kwargs = {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"}
        if idx == 0 and reply_to:
            kwargs["reply_to_message_id"] = reply_to
        try:
            bot.send_message(**kwargs)
        except telebot.apihelper.ApiTelegramException as e:
            log.error("Telegram send error: %s", e)
            bot.send_message(chat_id=chat_id, text=chunk)


def _edit_or_send(chat_id: int, message_id: int, text: str) -> None:
    try:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode="HTML",
        )
    except Exception as e:
        log.warning("edit_message_text failed (%s), sending new message.", e)
        _send_long(chat_id, text)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  FOLLOW-UP DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

_FOLLOWUP_PHRASES = [
    "i didn't understand", "i don't understand", "didn't get",
    "don't get", "explain again", "explain step", "what do you mean",
    "can you explain", "not clear", "confusing", "confused",
    "elaborate", "more detail", "step 2", "step 3", "which step",
    "why did you", "how did you get", "समझ नहीं", "फिर से समझाओ",
    "புரியவில்லை", "అర్థం కాలేదు", "মুঝে সমझ",
]


def _is_followup(text: str, chat_id: int) -> bool:
    """
    Returns True if this looks like a follow-up to the previous answer
    AND we have a previous answer stored for this chat.
    """
    if chat_id not in _last_qa:
        return False
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in _FOLLOWUP_PHRASES)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  LANGUAGE KEYBOARD
# ──────────────────────────────────────────────────────────────────────────────

def _language_keyboard() -> telebot.types.InlineKeyboardMarkup:
    kb = telebot.types.InlineKeyboardMarkup(row_width=3)
    buttons = [
        telebot.types.InlineKeyboardButton(
            text=name, callback_data=f"lang:{key}"
        )
        for key, name in SUPPORTED_LANGUAGES.items()
    ]
    kb.add(*buttons)
    kb.add(telebot.types.InlineKeyboardButton(
        text="🔄 Auto-detect", callback_data="lang:auto"
    ))
    return kb


# ──────────────────────────────────────────────────────────────────────────────
# 8.  COMMAND HANDLERS
# ──────────────────────────────────────────────────────────────────────────────

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    greeting = (
        "👋 <b>Welcome to LearnClear!</b>\n\n"
        "I am your AI tutor for Classes 1–12 — powered by Google Gemini.\n\n"
        "🌐 <b>Please choose your preferred language to get started:</b>"
    )
    bot.send_message(
        message.chat.id,
        greeting,
        parse_mode="HTML",
        reply_markup=_language_keyboard(),
    )


_welcomed: set[int] = set()


@bot.callback_query_handler(func=lambda call: call.data.startswith("lang:"))
def handle_language_callback(call):
    choice = call.data.split(":", 1)[1]
    chat_id = call.message.chat.id

    if choice == "auto":
        _user_lang.pop(chat_id, None)
        lang_name = "Auto-detect"
    elif choice in SUPPORTED_LANGUAGES:
        lang_name = SUPPORTED_LANGUAGES[choice]
        _user_lang[chat_id] = lang_name
    else:
        bot.answer_callback_query(call.id, "Unknown option.")
        return

    try:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=call.message.message_id,
            text=f"🌐 Language set to <b>{lang_name}</b>. You can change it any time with /language.",
            parse_mode="HTML",
        )
    except Exception:
        pass

    bot.answer_callback_query(call.id, f"✅ {lang_name} selected")

    if chat_id not in _welcomed:
        _welcomed.add(chat_id)
        how_to = (
            "✅ <b>All set!</b> Here is how to use LearnClear:\n\n"
            "📸 <b>Send a photo</b> of any question from your textbook or exam paper\n"
            "📄 <b>Upload a PDF</b> — worksheets and question papers work too\n"
            "💬 <b>Type your doubt</b> directly — any subject, any topic\n\n"
            "I support all subjects for Classes 1–12:\n"
            "Maths • Physics • Chemistry • Biology • History\n"
            "Geography • Civics • English • Hindi • Economics\n\n"
            "I will explain concepts step by step, show full working,\n"
            "give real-life examples, and suggest exam tips.\n\n"
            "📌 Commands:\n"
            "/language — change language\n"
            "/clear — forget previous question (start fresh)\n"
            "/subjects — see all supported subjects\n\n"
            "<i>⚠️ LearnClear is an AI tutor. Always verify important "
            "answers with your textbook or teacher before an exam.</i>"
        )
        bot.send_message(chat_id, how_to, parse_mode="HTML")


@bot.message_handler(commands=["language", "lang"])
def set_language(message):
    current = _get_lang(message.chat.id)
    current_label = current if current != "auto" else "Auto-detect"
    bot.reply_to(
        message,
        f"🌐 <b>Language Settings</b>\n\nCurrent: <b>{current_label}</b>\n\nTap a language to switch:",
        parse_mode="HTML",
        reply_markup=_language_keyboard(),
    )


@bot.message_handler(commands=["clear"])
def clear_history(message):
    _last_qa.pop(message.chat.id, None)
    bot.reply_to(
        message,
        "🗑 <b>Cleared!</b> Previous question forgotten. Send your next question.",
        parse_mode="HTML"
    )


@bot.message_handler(commands=["subjects"])
def show_subjects(message):
    subjects_text = (
        "📚 <b>Subjects I can help with (Classes 1–12):</b>\n\n"
        "🔢 Mathematics\n"
        "⚛️ Physics\n"
        "🧪 Chemistry\n"
        "🌿 Biology\n"
        "📖 History\n"
        "🗺 Geography\n"
        "🏛 Civics / Political Science\n"
        "✍️ English\n"
        "🇮🇳 Hindi / Sanskrit\n"
        "📊 Economics / Accountancy\n"
        "💻 Computer Science\n"
        "🌍 Environmental Science\n\n"
        "<i>Send a photo of your question or type your doubt to get started.</i>"
    )
    bot.reply_to(message, subjects_text, parse_mode="HTML")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  PHOTO HANDLER
# ──────────────────────────────────────────────────────────────────────────────

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    status = bot.reply_to(message, "🔍 Reading your question… please wait.")
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        img_bytes = bot.download_file(file_info.file_path)
        img_part  = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

        caption   = message.caption or ""
        lang_rule = _lang_instruction(message.chat.id, sample_text=caption)
        prompt    = build_learnclear_image_prompt(lang_rule)

        raw  = _gemini_generate([prompt, img_part])
        raw  = check_confidence(raw)
        text = sanitize_for_telegram(raw)

        # Store for potential follow-up
        _last_qa[message.chat.id] = {
            "question": caption or "[photo question]",
            "answer"  : raw
        }

        _edit_or_send(message.chat.id, status.message_id, text)

    except Exception as exc:
        log.error("handle_photo error: %s", exc)
        _edit_or_send(
            message.chat.id, status.message_id,
            "❌ I had trouble reading that image. Please make sure the question "
            "is clear and well-lit, then try again."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 10.  DOCUMENT HANDLER (PDF + image as file)
# ──────────────────────────────────────────────────────────────────────────────

_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


@bot.message_handler(content_types=["document"])
def handle_document(message):
    doc       = message.document
    mime_type = doc.mime_type or ""

    # ── Image sent as file ───────────────────────────────────────────────────
    if mime_type in _IMAGE_MIME_TYPES:
        status = bot.reply_to(message, "🔍 Reading your question… please wait.")
        try:
            file_info = bot.get_file(doc.file_id)
            img_bytes = bot.download_file(file_info.file_path)
            img_part  = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

            caption   = message.caption or ""
            lang_rule = _lang_instruction(message.chat.id, sample_text=caption)
            prompt    = build_learnclear_image_prompt(lang_rule)

            raw  = _gemini_generate([prompt, img_part])
            raw  = check_confidence(raw)
            text = sanitize_for_telegram(raw)

            _last_qa[message.chat.id] = {
                "question": caption or "[image question]",
                "answer"  : raw
            }
            _edit_or_send(message.chat.id, status.message_id, text)
        except Exception as exc:
            log.error("handle_document (image) error: %s", exc)
            _edit_or_send(
                message.chat.id, status.message_id,
                "❌ Could not read that image. Please try again."
            )
        return

    # ── PDF ──────────────────────────────────────────────────────────────────
    if mime_type == "application/pdf":
        status = bot.reply_to(message, "📄 Reading your question paper… please wait.")
        try:
            file_info = bot.get_file(doc.file_id)
            pdf_bytes = bot.download_file(file_info.file_path)
            pdf_part  = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

            caption   = message.caption or ""
            lang_rule = _lang_instruction(message.chat.id, sample_text=caption)
            prompt    = build_learnclear_image_prompt(lang_rule)

            raw  = _gemini_generate([prompt, pdf_part])
            raw  = check_confidence(raw)
            text = sanitize_for_telegram(raw)

            _last_qa[message.chat.id] = {
                "question": caption or "[PDF question]",
                "answer"  : raw
            }
            _edit_or_send(message.chat.id, status.message_id, text)
        except Exception as exc:
            log.error("handle_document (PDF) error: %s", exc)
            _edit_or_send(
                message.chat.id, status.message_id,
                "❌ I had trouble reading that PDF.\n\n"
                "Possible reasons:\n"
                "• The file is password-protected\n"
                "• The scan quality is very low\n\n"
                "Try sending a clear <b>photo</b> of the question instead."
            )
        return

    # ── Unsupported ──────────────────────────────────────────────────────────
    bot.reply_to(
        message,
        "⚠️ Please send your question as a <b>photo</b> or <b>PDF</b>.",
        parse_mode="HTML",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 11.  TEXT HANDLER — typed doubts + follow-ups
# ──────────────────────────────────────────────────────────────────────────────

@bot.message_handler(func=lambda m: m.content_type == "text")
def handle_text(message):
    user_text = message.text.strip()
    if not user_text or user_text.startswith("/"):
        return

    chat_id   = message.chat.id
    lang_rule = _lang_instruction(chat_id, sample_text=user_text)
    status    = bot.reply_to(message, "💬 Looking that up for you…")

    try:
        # ── Detect follow-up ─────────────────────────────────────────────────
        if _is_followup(user_text, chat_id):
            prev = _last_qa[chat_id]
            prompt = build_learnclear_followup_prompt(
                lang_rule=lang_rule,
                original_question=prev["question"],
                bot_answer=prev["answer"],
                followup=user_text,
            )
            log.info("Follow-up detected for chat %d", chat_id)
        else:
            prompt = build_learnclear_text_prompt(lang_rule, user_text)

        raw  = _gemini_generate([prompt])
        raw  = check_confidence(raw)
        text = sanitize_for_telegram(raw)

        # Update history with latest Q&A
        _last_qa[chat_id] = {
            "question": user_text,
            "answer"  : raw
        }

        _edit_or_send(chat_id, status.message_id, text)

    except Exception as exc:
        log.error("handle_text error: %s", exc)
        _edit_or_send(
            chat_id, status.message_id,
            "❌ I couldn't process that question. Please rephrase and try again."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 12.  FLASK SERVER (keep-alive + wake endpoint)
# ──────────────────────────────────────────────────────────────────────────────

server = Flask(__name__)


@server.route("/")
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LearnClear | AI Tutor for Indian Students</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      min-height: 100vh;
      display: flex; align-items: center; justify-content: center;
      background: linear-gradient(135deg, #e8f5e9 0%, #e3f2fd 100%);
      color: #1e293b; padding: 20px;
    }
    .card {
      background: #ffffff; border-radius: 24px;
      box-shadow: 0 25px 50px -12px rgba(0,0,0,0.12);
      max-width: 480px; width: 100%;
      padding: 48px 36px 40px; text-align: center;
    }
    .logo {
      width: 88px; height: 88px;
      background: linear-gradient(135deg, #1565C0, #00ACC1);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 42px; margin: 0 auto 24px;
      box-shadow: 0 8px 24px rgba(21,101,192,0.3);
    }
    h1 { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 10px; }
    .tagline { font-size: 1.05rem; color: #64748b; line-height: 1.6; margin-bottom: 32px; }
    .features {
      background: #f8fafc; border-radius: 16px;
      padding: 20px 24px; text-align: left; margin-bottom: 32px;
    }
    .features h2 {
      font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: .08em; color: #94a3b8; margin-bottom: 14px;
    }
    .feature {
      display: flex; align-items: center; gap: 12px;
      font-size: 0.95rem; color: #334155; padding: 6px 0;
    }
    .feature-icon { font-size: 1.2rem; flex-shrink: 0; }
    .btn {
      display: block;
      background: linear-gradient(135deg, #1565C0, #00ACC1);
      color: #ffffff; text-decoration: none;
      padding: 18px 32px; border-radius: 14px;
      font-weight: 700; font-size: 1.05rem;
      transition: transform 0.18s, box-shadow 0.18s;
      box-shadow: 0 6px 20px rgba(21,101,192,0.35);
    }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(21,101,192,0.4); }
    .disclaimer {
      margin-top: 28px; padding-top: 22px;
      border-top: 1px solid #f1f5f9;
      font-size: 0.72rem; color: #94a3b8; line-height: 1.5;
    }
    .status-dot {
      display: inline-block; width: 8px; height: 8px;
      background: #22c55e; border-radius: 50%;
      animation: pulse 2s infinite;
      margin-right: 6px; vertical-align: middle;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">📚</div>
    <h1>LearnClear</h1>
    <p class="tagline">Your AI tutor for Classes 1–12 — explains concepts, solves problems, answers doubts in your own language.</p>
    <div class="features">
      <h2>What I can do</h2>
      <div class="feature"><span class="feature-icon">📸</span> Solve questions from photos of textbooks &amp; papers</div>
      <div class="feature"><span class="feature-icon">📄</span> Process PDF worksheets and question papers</div>
      <div class="feature"><span class="feature-icon">🔢</span> Step-by-step working for Maths &amp; Science</div>
      <div class="feature"><span class="feature-icon">💡</span> Memory tips and exam angles for every answer</div>
      <div class="feature"><span class="feature-icon">🌐</span> Responds in 12 Indian languages</div>
      <div class="feature"><span class="feature-icon">📚</span> All CBSE, ICSE &amp; State Board subjects</div>
    </div>
    <a href="https://t.me/learn_clear_bot?start=welcome" class="btn">🚀 Open LearnClear on Telegram</a>
    <p class="disclaimer">
      <span class="status-dot"></span><strong>Bot is online</strong><br><br>
      <strong>Note:</strong> LearnClear is an AI tutor for guidance only.
      Always verify important answers with your textbook or teacher before an exam.
    </p>
  </div>
</body>
</html>"""


@server.route("/health")
def health():
    """
    Required by Cloud Run — must return 200 for traffic to be routed here.
    Cloud Run checks this endpoint to confirm the container is healthy.
    """
    return {"status": "ok", "service": "learn-clear"}, 200


@server.route("/wake")
def wake():
    """Optional manual check endpoint."""
    return {"status": "awake", "service": "learn-clear"}, 200


def _run_flask():
    port = int(os.environ.get("PORT", 8080))
    log.info("Flask starting on port %d", port)
    server.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False,
        threaded=True
    )


def keep_alive():
    t = threading.Thread(target=_run_flask, name="flask-keepalive", daemon=True)
    t.start()
    log.info("Flask keep-alive started.")


# ──────────────────────────────────────────────────────────────────────────────
# 13.  POLLING — simple thread, no watchdog needed on Cloud Run
# ──────────────────────────────────────────────────────────────────────────────
# Cloud Run with --min-instances 1 keeps the container alive 24/7.
# No cold starts, no idle shutdowns, no UptimeRobot required.
# A single daemon thread handles Telegram polling reliably.

def _start_polling():
    """Start infinity_polling in a background daemon thread."""
    def _poll():
        log.info("Polling thread started.")
        bot.infinity_polling(
            none_stop=True,
            interval=0,
            timeout=20,
            long_polling_timeout=20,
            logger_level=logging.WARNING,
            allowed_updates=["message", "callback_query"],
        )
    t = threading.Thread(target=_poll, name="bot-polling", daemon=True)
    t.start()


# ──────────────────────────────────────────────────────────────────────────────
# 14.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start Flask FIRST and wait for it to bind to port
    # Cloud Run health check hits port 8080 within seconds of startup
    keep_alive()
    time.sleep(3)   # Give Flask time to bind to port 8080

    # Then start Telegram polling
    _start_polling()

    log.info("LearnClear is live on Google Cloud Run.")

    while True:
        time.sleep(60)