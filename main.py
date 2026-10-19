
import os
import asyncio
import logging
import threading

from flask import Flask, jsonify
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from google import genai

# ---------------- CONFIGURATION ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are a helpful, accurate AI assistant.
Answer the user's actual question clearly.
Never invent facts, sources, or search results.
If you are uncertain, say so honestly.
Use the language requested by the user.
Explain technical topics step by step when useful.
"""

# ---------------- HEALTH SERVER ----------------

web = Flask(__name__)


@web.get("/")
def home():
    return jsonify({
        "service": "Gemini Telegram Bot",
        "status": "online"
    })


@web.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


def run_web_server():
    port = int(os.environ.get("PORT", "10000"))
    web.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False,
    )


# ---------------- TELEGRAM COMMANDS ----------------

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.effective_message.reply_text(
        "👋 Hello! I'm your Gemini AI chatbot.\n\n"
        "Send me a message to start chatting.\n"
        "/start - Start the bot\n"
        "/help - Get help\n"
        "/reset - Clear conversation context"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.effective_message.reply_text(
        "💡 Send a text message and I'll ask Gemini to answer it.\n"
        "Use /reset to clear the current conversation."
    )


async def reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    context.user_data.pop("history", None)
    await update.effective_message.reply_text(
        "🧹 Conversation context cleared!"
    )


# ---------------- GEMINI CHAT ----------------

async def chat(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.effective_message

    if message is None or not message.text:
        return

    user_text = message.text.strip()

    if not user_text:
        return

    if len(user_text) > 12000:
        await message.reply_text(
            "Please send a shorter message (maximum 12,000 characters)."
        )
        return

    await context.bot.send_chat_action(
        chat_id=message.chat_id,
        action=ChatAction.TYPING,
    )

    history = context.user_data.setdefault("history", [])

    # Keep a short conversation history.
    history.append(("user", user_text))
    history = history[-12:]
    context.user_data["history"] = history

    conversation = "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {text}"
        for role, text in history
    )

    prompt = f"{SYSTEM_PROMPT}\n\nConversation:\n{conversation}\nAssistant:"

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer = response.text

        if not answer or not answer.strip():
            logging.error("Gemini returned an empty response")
            await message.reply_text(
                "Gemini returned an empty response. Please try again."
            )
            return

        answer = answer.strip()
        history.append(("assistant", answer))
        context.user_data["history"] = history[-12:]

        # Telegram messages must fit within its message limit.
        chunks = [
            answer[i:i + 4000]
            for i in range(0, len(answer), 4000)
        ]

        for chunk in chunks:
            await message.reply_text(chunk)

    except Exception as exc:
        # Detailed diagnostic is recorded in Render logs.
        logging.exception(
            "Gemini request failed: %s",
            type(exc).__name__,
        )

        await message.reply_text(
            "⚠️ Gemini is temporarily unavailable.\n\n"
            "Please try again shortly. If the problem persists, "
            "check the Render logs for the exact error."
        )


# ---------------- START BOT ----------------

def main():
    logging.info("Starting Gemini Telegram Bot")

    threading.Thread(
        target=run_web_server,
        daemon=True,
    ).start()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat)
    )

    logging.info("Telegram polling is starting")

    application.run_polling(
        drop_pending_updates=False,
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    main()
  
