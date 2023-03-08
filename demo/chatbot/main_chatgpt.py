"""
A ChatGPT demo, built in Meerkat.
"""
import os

import meerkat as mk
from demo.chatbot.chatbot_chatgpt import ChatGPT

FILEPATH = os.path.dirname(os.path.abspath(__file__))

chatgpt = ChatGPT(savepath=os.path.join(FILEPATH, "history.chatgpt.jsonl"))
page = mk.gui.Page(chatgpt, id="chatgpt")
page.launch()
