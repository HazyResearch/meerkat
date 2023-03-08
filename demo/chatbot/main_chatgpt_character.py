"""A lofi version of character.ai, built in Meerkat with ChatGPT."""
import os

import meerkat as mk
from demo.chatbot.chatbot_chatgpt_character import CharacterChatbot

FILEPATH = os.path.dirname(os.path.abspath(__file__))

charbot = CharacterChatbot(
    savepath=os.path.join(FILEPATH, "history.charbot.jsonl"),
)
page = mk.gui.Page(charbot, id="charbot")
page.launch()
