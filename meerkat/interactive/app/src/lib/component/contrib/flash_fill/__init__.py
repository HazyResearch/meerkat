import re
from functools import partial
from typing import TYPE_CHECKING, Dict

from manifest import Manifest

from ...html import div

if TYPE_CHECKING:
    from meerkat import DataFrame, Store, Component


class FlashFill(div):
    def __init__(
        self,
        df: "DataFrame",
        target_column: str,
        classes: str = "",
    ):
        df = df.view()
        if target_column not in df.columns:
            df[target_column] = ""
        component = _build_component(df)
        super().__init__(
            slots=[component],
            classes=classes,
        )

    def _get_ipython_height(self):
        return "600px"


def _build_component(df: "DataFrame") -> "Component":
    import meerkat as mk

    def complete_prompt(row: Dict[str, any], example_template: mk.Store[str]):
        assert isinstance(row, dict)
        output = example_template.format(**row)
        return output

    @mk.reactive
    def format_output_col(output_col):
        if output_col is None:
            return {
                "text": "Template doesn't end with column.",
                "classes": "text-red-600 px-2 text-sm",
            }
        else:
            return {
                "text": output_col,
                "classes": "font-mono font-bold bg-slate-200 rounded-md px-2 text-violet-600 w-fit",
            }

    @mk.endpoint
    def on_edit(df: mk.DataFrame, column: str, keyidx: any, posidx: int, value: any):
        df.loc[keyidx, column] = value

    @mk.reactive()
    def update_df_with_example_template(df: mk.DataFrame, template: mk.Store[str]):
        """Update the df with the new prompt template.

        This is as simple as returning a view.
        """
        df = df.view()

        # Extract out the example template from the prompt template.
        df["example"] = mk.defer(
            df,
            function=partial(complete_prompt, example_template=template),
            inputs="row",
        )
        return df

    @mk.reactive()
    def check_example_template(example_template: str, df: mk.DataFrame):
        example_template = example_template.strip()
        # check if example_template ends with {.*} using a regex and extract the content
        # between the brackets
        # Define the regular expression pattern
        pattern = r"\{([^}]+)\}$"

        # Use re.search to find the match in the example_template string
        match = re.search(pattern, example_template)

        if match:
            # Extract the content between the curly braces using the group() method
            content = match.group(1)

            if content not in df.columns:
                raise ValueError(
                    f"The column '{content}' does not exist in the dataframe."
                )
        else:
            return None
            # raise ValueError("The example template must end with '{column_name}'")
        return content

    @mk.endpoint()
    def run_manifest(
        instruct_cmd: str, df: mk.DataFrame, output_col: str, selected: list, api: str
    ):
        client_name, engine = api.split("/")
        manifest = Manifest(
            client_name=client_name,
            client_connection=open("/Users/sabrieyuboglu/.meerkat/keys/.openai").read(),
            engine=engine,
            temperature=0,
            max_tokens=1,
        )

        def _run_manifest(example: mk.Column):
            # Format instruct-example-instruct prompt.
            out = manifest.run(
                [f"{instruct_cmd} {in_context_examples} {x}" for x in example]
            )
            return out

        selected_idxs = df.primary_key.isin(selected)

        # Concat all of the in-context examples.
        train_df = df[(~selected_idxs) & (df[output_col] != "")]
        in_context_examples = "\n".join(train_df["example"]())

        fill_df = df[selected_idxs]

        # Filter based on train/abstain/fill
        # Only fill in the blanks.
        col = output_col
        flash_fill = mk.map(
            fill_df, function=_run_manifest, is_batched_fn=True, batch_size=4
        )

        # If the dataframe does not have the column, add it.
        if col not in df.columns:
            df[col] = ""

        df[col][selected_idxs] = flash_fill
        df.set(df)
        print("done with manifest")

    df = df.mark()

    instruction_editor = mk.gui.Editor(
        code="", title="Instruction Editor"  # Is this paper theoretical or empirical?",
    )
    example_template_editor = mk.gui.Editor(
        code="",  # "Abstract: {abstract}, Title: {title}; Answer: {answer}",
        title="Training Template Editor",
    )

    example_template = example_template_editor.code

    output_col = check_example_template(example_template=example_template, df=df)

    df_view = update_df_with_example_template(df, example_template)

    table = mk.gui.Table(
        df_view,
        on_edit=on_edit.partial(df=df),
    )

    api_select = mk.gui.core.Select(
        values=[
            "together/gpt-j-6b",
            "together/gpt-2",
            "openai/text-davinci-003",
            "openai/code-davinci-002",
        ],
        value="together/gpt-j-6b"
    )

    run_manifest_button = mk.gui.Button(
        title="Flash Fill",
        icon="Magic",
        on_click=run_manifest.partial(
            instruct_cmd=instruction_editor.code,
            df=df_view,
            output_col=output_col,
            selected=table.selected,
            api=api_select,
        ),
    )
    formatted_output_col = format_output_col(output_col)

    overview_panel = mk.gui.html.flexcol(
        [
            mk.gui.Text(
                "Infer selected rows using in-context learning.",
                classes="font-bold text-slate-600 text-sm",
            ),
            mk.gui.Text(
                "Specify the instruction and a template for in-context examples.",
                classes="text-slate-600 text-sm",
            ),
            mk.gui.html.div(
                [
                    mk.gui.Text("Target column: ", classes="text-slate-600 text-sm"),
                    mk.gui.Text(
                        formatted_output_col["text"],
                        classes=formatted_output_col["classes"],  # noqa: E501
                    ),
                ],
                classes="gap-3 align-middle grid grid-cols-[auto_1fr]",
            ),
            mk.gui.html.grid(
                [
                    run_manifest_button,
                    api_select,
                ],
                classes="grid grid-cols-[auto_1fr] gap-2",
            ),
        ],
        classes="items-left mx-4 gap-1",
    )
    prompt_editor = mk.gui.html.flexcol(
        [
            instruction_editor,
            example_template_editor,
        ],
        classes="flex-1 gap-1",
    )
    return mk.gui.html.div(
        [
            mk.gui.html.grid(
                [overview_panel, prompt_editor],
                classes="grid grid-cols-[1fr_3fr] space-x-5",
            ),
            mk.gui.html.div([table], classes="h-full w-screen"),
        ],
        classes="gap-4 h-screen grid grid-rows-[auto_1fr]",
    )
