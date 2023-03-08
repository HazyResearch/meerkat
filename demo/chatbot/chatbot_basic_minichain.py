"""
A basic chatbot implementation that uses Meerkat to 
create a chatbot and manage its state, and MiniChain to
perform prompting.

Borrows code from [MiniChain](https://github.com/srush/minichain)'s
chatbot example.
"""
import json
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import jinja2
import minichain
import rich

import meerkat as mk

CHATBOT = "chatbot"
USER = "user"


@dataclass
class Memory:
    """Generic stateful memory."""

    memory: List[Tuple[str, str]]
    size: int = 2
    human_input: str = ""

    def push(self, response: str) -> "Memory":
        memory = self.memory if len(self.memory) < self.size else self.memory[1:]
        self.memory = memory + [(self.human_input, response)]
        return self


# Get path to this file
PATH = pathlib.Path(__file__).parent.absolute()


class ChatPrompt(minichain.TemplatePrompt):
    """Chat prompt with memory."""

    template = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(PATH))
    ).get_template("chatbot.pmpt.tpl")

    def parse(self, out: str, inp: Memory) -> Memory:
        result = out.split("Assistant:")[-1]
        return inp.push(result)


class ConversationHistory:
    """Stores the full conversation history, and keeps track of
    the agent's memory to use for prompting."""

    def __init__(
        self,
        greeting: str = "Hi! Welcome to Meerkat!",
        chatbot_name: str = "Meerkat",
        savepath: str = "history.jsonl",
    ):
        # Create an data frame with a single message from the chatbot.
        df = mk.DataFrame(
            {
                "message": [greeting],
                "sender": [CHATBOT],
                "name": [chatbot_name],
                "time": [self.timestamp(datetime.now())],
            },
        )
        self.df = df

        # Create a memory component that will be used to do prompting.
        self.memory = Memory([], size=2)

        # Store the history in a jsonl file.
        self.savepath = savepath
        if os.path.exists(savepath):
            self.df = mk.DataFrame.from_json(savepath, lines=True)
            self.restore_memory()
        else:
            self.write_last_message()

    def restore_memory(self):
        # Restore the memory by looking back at the last
        # 2 * memory.size messages (if possible)
        if len(self.df) < 2 * self.memory.size:
            lookback = (len(self.df) // 2) * 2
        else:
            lookback = self.memory.size * 2
        window = self.df[-lookback:]
        self.memory = Memory(
            [
                (window[i]["message"], window[i + 1]["message"])
                for i in range(0, len(window), 2)
            ]
        )

    @staticmethod
    def timestamp(time: datetime):
        # Formats as 04:20 PM / Jan 01, 2020
        return time.strftime("%I:%M %p / %b %d, %Y")

    def update(self, message: str, sender: str, name: str, send_time: datetime):
        df = mk.DataFrame(
            {
                "message": [message],
                "sender": [sender],
                "name": [name],
                "time": [self.timestamp(send_time)],
            },
        )

        # THIS IS ESSENTIAL!
        # Running a df.set will automatically trigger a re-render on the
        # frontend, AS LONG AS this method is called inside an `mk.endpoint`!
        # Otherwise, this will behave like a normal Python method.
        self.df.set(self.df.append(df))

        self.write_last_message()

    def write_last_message(self):
        # Write the last message.
        with open(self.savepath, "a") as f:
            f.write(json.dumps(self.df[-1]) + "\n")


def get_chatbot_response(prompt: ChatPrompt, memory: Memory, message: str) -> str:
    """Run the minichain prompt, update the state, and get the response."""
    memory.human_input = message
    memory = prompt(memory)
    response = memory.memory[-1][1]
    return response


@mk.endpoint()
def run_chatbot(
    history: "ConversationHistory",
    message: str,
    prompt: ChatPrompt,
    chatbot_name: str = "Meerkat",
    user_name: str = "Me",
):
    """Update the conversation history and get a response from the chatbot."""
    history.update(message, USER, user_name, datetime.now())
    response = get_chatbot_response(prompt, history.memory, message)
    history.update(response, CHATBOT, chatbot_name, datetime.now())


# A pure Python component in Meerkat just needs to subclass `div`!
class BasicChatbot(mk.gui.html.div):
    """A basic chatbot component."""

    # Expose the chat Component that is used internally
    chat: mk.gui.Chat = None

    # Expose the chat history
    history: ConversationHistory = None

    # Expose the `on_send` endpoint of the `Chat` component.
    on_send: mk.gui.Endpoint = None

    def __init__(
        self,
        model: str = "curie",
        chatbot_name: str = "🤖 Meerkat",
        user_name: str = "Me",
        img_chatbot: str = "http://meerkat.wiki/favicon.png",
        img_user: str = "http://placekitten.com/200/200",
        savepath: str = "history.jsonl",
        classes: str = "h-full flex flex-col",
    ):

        # Spin up the prompt engine with `minichain`.
        backend = minichain.start_chain("chatbot")
        prompt = ChatPrompt(backend.OpenAI(model=model))

        # Keep track of the conversation history.
        # Also contains a memory that is used for prompting.
        history = ConversationHistory(
            chatbot_name=chatbot_name,
            savepath=savepath,
        )

        # This endpoint takes in one remaining argument, `message`, which
        # is the message sent by the user. It then updates the conversation
        # history, and gets a response from the chatbot.
        on_send: mk.gui.Endpoint = run_chatbot.partial(
            history=history,
            prompt=prompt,
            chatbot_name=chatbot_name,
            user_name=user_name,
        )

        # Create the chat component
        chat = mk.gui.Chat(
            df=history.df,
            img_chatbot=img_chatbot,
            img_user=img_user,
            on_send=on_send,
        )

        # Make a little header on top of the chat component.
        header = mk.gui.html.div(
            mk.gui.Caption(
                f"Chatbot running `{model}`, and built using 🔮 [Meerkat](http://meerkat.wiki)."
            ),
            classes="bg-gray-50 dark:bg-slate-300 mx-8 px-2 w-fit rounded-t-lg",
        )

        # Call the constructor for `div`, which we wrap around
        # any components made here.
        super().__init__([header, chat], classes=classes)

        # Store stuff
        self.chat = chat
        self.history = history
        self.on_send = on_send


def test_basic_chatbot():
    """Test the basic chatbot component by simulating a conversation
    and checking that the memory is updated."""

    # Create a basic chatbot
    chatbot = BasicChatbot("curie")

    # Simulate a conversation and check the history.
    chatbot.on_send.run(message="Who are you?")
    rich.print(chatbot.history.memory)
    chatbot.on_send.run(message="How are you?")
    rich.print(chatbot.history.memory)
    chatbot.on_send.run(message="What is your name?")
    rich.print(chatbot.history.memory)
