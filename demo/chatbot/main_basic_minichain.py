"""
A basic chatbot demo, powered by Meerkat and Minichain.
"""
import os

import meerkat as mk
from demo.chatbot.chatbot_basic_minichain import BasicChatbot

FILEPATH = os.path.dirname(os.path.abspath(__file__))

chatbot = BasicChatbot(
    model="text-davinci-003",
    chatbot_name="🤖 Meerkat",
    user_name="Me",
    img_chatbot="http://meerkat.wiki/favicon.png",
    img_user="http://placekitten.com/200/200",
    savepath=os.path.join(FILEPATH, "history.jsonl"),
)
page = mk.gui.Page(chatbot, id="chatbot")
page.launch()
