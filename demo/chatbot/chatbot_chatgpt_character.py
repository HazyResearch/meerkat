import json
import os
import shutil
from datetime import datetime

import openai

import meerkat as mk
from demo.chatbot.characters import CHARACTERS

CHATBOT = "chatbot"
USER = "user"


class ConversationHistory:
    """Stores the full conversation history, and keeps track of the agent's
    memory to use for prompting."""

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

        # Store the history in a jsonl file.
        self.savepath = savepath
        if os.path.exists(savepath):
            self.df = mk.DataFrame.from_json(savepath, lines=True)
        else:
            self.write_last_message()

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


def get_chatbot_response(
    history: ConversationHistory,
    lookback: int,
    instructions: str = "You are a helpful assistant.",
) -> str:
    """Run the OpenAI chat completion, using a subset of the chat history."""
    assert lookback > 0, "Lookback must be greater than 0."

    # Lookback, and rename columns to align with OpenAI's API.
    messages = history.df[-lookback:].rename({"sender": "role", "message": "content"})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": instructions}]
        + [
            message if message["role"] == USER else {**message, **{"role": "assistant"}}
            for message in messages["role", "content"]
        ],
    )
    return response["choices"][0]["message"]["content"]


@mk.endpoint()
def run_chatbot(
    history: ConversationHistory,
    message: str,
    lookback: int = 10,
    instructions: str = "You are a helpful assistant.",
    chatbot_name: str = "Meerkat",
    user_name: str = "Me",
):
    """Update the conversation history and get a response from the chatbot."""
    history.update(message, USER, user_name, datetime.now())
    response = get_chatbot_response(history, lookback, instructions)
    history.update(response, CHATBOT, chatbot_name, datetime.now())


# A pure Python component in Meerkat just needs to subclass `div`!
class CharacterChatbot(mk.gui.html.div):
    """A basic chatbot component."""

    # Expose the chat Component that is used internally
    chat: mk.gui.Chat = None

    # Expose the chat history
    history: ConversationHistory = None

    # Expose the `on_send` endpoint of the `Chat` component.
    on_send: mk.gui.Endpoint = None

    def __init__(
        self,
        lookback: int = 10,
        user_name: str = "Me",
        img_user: str = "http://placekitten.com/200/200",
        savepath: str = "history.jsonl",
        classes: str = "h-full flex flex-col pb-12",
    ):
        # Selector for character name
        character_select = mk.gui.Select(
            values=list(range(len(CHARACTERS))),
            labels=[c["character"] for c in CHARACTERS],
            value=0,
        )
        character = mk.reactive(lambda value: CHARACTERS[value])(character_select.value)
        chatbot_name = mk.reactive(lambda character: "🤖 " + character["character"])(
            character
        )
        instructions = mk.reactive(
            lambda character: character["instructions"]
            + f" You must only answer as {character['character']},"
            " and never refer to yourself as a language model."
            " Make your responses as crazy and whimsical as possible."
        )(character)
        mk.gui.print(character)

        # Keep track of the conversation history.
        # Also contains a memory that is used for prompting.
        history = mk.reactive(
            lambda character: ConversationHistory(
                greeting=character["greeting"],
                chatbot_name=chatbot_name,
                savepath=savepath.replace(".jsonl", f".{character['id']}.jsonl"),
            )
        )(character)

        # This endpoint takes in one remaining argument, `message`, which
        # is the message sent by the user. It then updates the conversation
        # history, and gets a response from the chatbot.
        on_send: mk.gui.Endpoint = run_chatbot.partial(
            history=history,
            lookback=lookback,
            instructions=instructions,
            chatbot_name=chatbot_name,
            user_name=user_name,
        )

        # Create the chat component
        with mk.magic():
            chat = mk.gui.Chat(
                df=history.df,
                img_chatbot=character["img"],
                img_user=img_user,
                on_send=on_send,
            )

        # Make a little header on top of the chat component.
        header = mk.gui.Caption(
            "Character chat, built using 🔮 [Meerkat](http://meerkat.wiki).",
            classes="self-center max-w-full bg-gray-50 dark:bg-slate-300"
            " mx-8 px-2 w-fit text-center rounded-t-lg",
        )

        # Call the constructor for `div`, which we wrap around
        # any components made here.
        super().__init__([header, chat, character_select], classes=classes)

        # Store stuff
        self.chat = chat
        self.history = history
        self.on_send = on_send


def test_chatgpt():
    """Test the chatbot component by simulating a conversation."""

    # Create chatbot
    chatbot = CharacterChatbot(savepath="temp.history.jsonl")

    # Simulate a conversation and check the history.
    chatbot.on_send.run(message="Who are you?")
    chatbot.on_send.run(message="How are you?")
    chatbot.on_send.run(message="What is your name?")

    # Remove the temp file.
    shutil.rmtree("temp.history.jsonl", ignore_errors=True)
