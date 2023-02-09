import meerkat as mk


def _default_df():
    return mk.DataFrame(
        {
            "message": ["Lorem ipsum"],
            "sender": ["chatbot"],
            "name": ["ChatBot"],
            "time": ["2 hours ago"],
        }
    )


@mk.gui.endpoint
def on_send(df: mk.DataFrame, message: str):
    df.set(
        df.append(
            mk.DataFrame(
                {
                    "message": [message, "random message"],
                    "sender": ["user", "chatbot"],
                    "name": ["User", "ChatBot"],
                    "time": ["1 hour ago", "1 hour ago"],
                }
            )
        )
    )


def test_on_send():
    df = _default_df()
    chat = mk.gui.core.Chat(
        df=df,
        img_chatbot="https://placeimg.com/200/200/animals",
        img_user="https://placeimg.com/200/200/people",
        on_send=on_send.partial(df=df),
    )
    chat.on_send(message="hello")
    assert len(df) == 3
    assert df["message"][1] == "hello"
