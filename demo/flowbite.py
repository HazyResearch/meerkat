"""An assortment of Flowbite components."""
import meerkat as mk
from meerkat.interactive import flowbite as fb
from meerkat.interactive import html as html


@mk.endpoint()
def on_click():
    print("Clicked!")


button = fb.Button(
    slots="Click me",
    on_click=on_click,
)

card = fb.Card(
    slots=[
        mk.gui.html.flex(
            slots=[button, "My Custom Card"],
            classes="flex-row items-center justify-between",
        )
    ],
    color="red",
    rounded=True,
    size="lg",
)


accordion = fb.Accordion(
    slots=[
        fb.AccordionItem(
            slots=[
                html.span(slots="My Header 1"),
                html.p(
                    slots=[
                        "Lorem ipsum dolor sit amet, consectetur adipisicing elit. "
                        "Illo ab necessitatibus sint explicabo ..."
                        "Check out this guide to learn how to ",
                        html.a(
                            slots="get started",
                            href="https://tailwindcss.com/docs",
                            target="_blank",
                            rel="noreferrer",
                            classes="text-blue-600 dark:text-blue-500 hover:underline",
                        ),
                        " and start developing websites even faster with components "
                        "on top of Tailwind CSS.",  # noqa: E501
                    ],
                    classes="mb-2 text-gray-500 dark:text-gray-400",
                ),
            ]
        ),
        fb.AccordionItem(
            slots=[
                html.span(slots="My Header 2"),
                html.p(
                    slots=[
                        "Lorem ipsum dolor sit amet, consectetur adipisicing elit. "
                        "Illo ab necessitatibus sint explicabo ...",
                        "Lorem ipsum dolor sit amet, consectetur adipisicing elit. "
                        "Illo ab necessitatibus sint explicabo ...",
                        "Learn more about these technologies:",
                        html.ul(
                            slots=[
                                html.li(
                                    slots=html.a(
                                        slots="Lorem ipsum",
                                        href="https://tailwindcss.com/docs",
                                        target="_blank",
                                        rel="noreferrer",
                                        classes="text-blue-600 dark:text-blue-500 hover:underline",  # noqa: E501
                                    )
                                ),
                                html.li(
                                    slots=html.a(
                                        slots="Tailwind UI",
                                        href="https://tailwindui.com/",
                                        target="_blank",
                                        rel="noreferrer",
                                        classes="text-blue-600 dark:text-blue-500 hover:underline",  # noqa: E501
                                    )
                                ),
                            ],
                            classes="list-disc pl-5 dark:text-gray-400 text-gray-500",
                        ),
                    ],
                    classes="mb-2 text-gray-500 dark:text-gray-400",
                ),
            ]
        ),
    ]
)

alert = fb.Alert(
    slots=[
        html.span(
            slots=[
                html.svg(
                    slots=[
                        html.path(
                            fill_rule="evenodd",
                            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z",  # noqa: E501
                            clip_rule="evenodd",
                        )
                    ],
                    aria_hidden="true",
                    classes="w-5 h-5",
                    fill="currentColor",
                    viewBox="0 0 20 20",
                )
            ],
            # slot="icon",
        ),
        html.span(slots="This is a info alert"),
        html.div(
            slots=[
                html.div(
                    slots=[
                        "More info about this info alert goes here. This example text "
                        "is going to run a bit longer so that you can see how spacing "
                        "within an alert works with this kind of content.",
                    ],
                    classes="mt-2 mb-4 text-sm",
                ),
                html.div(
                    slots=[
                        fb.Button(
                            slots=[
                                html.svg(
                                    slots=[
                                        html.path(d="M10 12a2 2 0 100-4 2 2 0 000 4z"),
                                        html.path(
                                            d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z",  # noqa: E501
                                            fill_rule="evenodd",
                                            clip_rule="evenodd",
                                        ),
                                    ],
                                    classes="-ml-0.5 mr-2 h-4 w-4",
                                    fill="currentColor",
                                    viewBox="0 0 20 20",
                                    aria_hidden="true",
                                ),
                                "View more",
                            ],
                            size="xs",
                        ),
                        fb.Button(
                            slots="Go to Home",
                            size="xs",
                            outline=True,
                            color="blue",
                            btnClass="dark:!text-blue-800",
                        ),
                    ],
                    classes="flex gap-2",
                ),
            ],
            classes="mt-4",
        ),
    ],
    color="blue",
)

avatar = html.div(
    slots=[
        html.div(
            slots=[
                fb.Avatar(
                    src="http://placekitten.com/200/200",
                    stacked=True,
                    classes="mr-2",
                ),
                fb.Avatar(
                    src="http://placekitten.com/200/200",
                    stacked=True,
                    classes="mr-2",
                ),
                fb.Avatar(
                    src="http://placekitten.com/200/200",
                    stacked=True,
                    classes="mr-2",
                ),
                fb.Avatar(stacked=True),
            ],
            classes="flex mb-5",
        ),
        html.div(
            slots=[
                fb.Avatar(
                    src="http://placekitten.com/200/200",
                    stacked=True,
                    classes="mr-2",
                ),
                fb.Avatar(
                    src="http://placekitten.com/200/200",
                    stacked=True,
                    classes="mr-2",
                ),
                fb.Avatar(
                    src="http://placekitten.com/200/200",
                    stacked=True,
                    classes="mr-2",
                ),
                fb.Avatar(
                    stacked=True,
                    href="/",
                    classes="bg-gray-700 text-white hover:bg-gray-600 text-sm",
                    slots="+99",
                ),
            ],
            classes="flex",
        ),
    ],
    classes="flex flex-col",
)


carousel = html.div(
    slots=[
        fb.Carousel(
            images=[
                {
                    "id": "1",
                    "imgurl": "http://placeimg.com/640/480/nature",
                    "name": "Image 1",
                },
                {
                    "id": "2",
                    "imgurl": "http://placeimg.com/640/480/animals",
                    "name": "Image 2",
                },
                {
                    "id": "3",
                    "imgurl": "http://placeimg.com/640/480/tech",
                    "name": "Image 3",
                },
            ],
        )
    ],
    classes="max-w-4xl",
)

toast = fb.Toast(
    slots=[
        html.div(
            slots=[
                html.div(
                    slots=[
                        html.svg(
                            slots=[
                                html.path(
                                    d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z",  # noqa: E501
                                    stroke_linecap="round",
                                    stroke_linejoin="round",
                                    stroke_width="1.5",
                                    stroke="currentColor",
                                    fill="none",
                                    classes="w-6 h-6",
                                ),
                            ],
                            viewBox="0 0 24 24",
                        ),
                    ],
                    classes="flex-shrink-0",
                ),
                html.div(
                    slots="Default color is blue.",
                    classes="ml-3",
                ),
            ],
            classes="flex",
        ),
    ],
    classes="mb-2",
    open=True,
    position="top-left",
)


heading = html.h1(slots="Flowbite Demo", classes="text-4xl")

page = mk.gui.Page(
    mk.gui.html.flexcol([heading, card, accordion, alert, avatar, carousel, toast]),
    id="flowbite",
)
page.launch()
