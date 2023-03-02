import os
import re
from functools import partial
from typing import TYPE_CHECKING, Dict

from ...html import div

if TYPE_CHECKING:
    from meerkat import Component, DataFrame


class FlashFill(div):
    """A component for flash filling a column of data using large language
    models.

    Args:
        df (DataFrame): The dataframe to flash fill.
        target_column (str): The column to flash fill.
    """

    def __init__(
        self,
        df: "DataFrame",
        target_column: str,
        manifest_cache_dir: str = "~/.cache/manifest",
    ):
        df = df.view()
        if target_column not in df.columns:
            df[target_column] = ""
        df["is_train"] = False
        component, prompt = self._build_component(df)
        super().__init__(
            slots=[component],
            classes="h-full w-full",
        )
        self._prompt = prompt
        self.manifest_cache_dir = os.path.abspath(
            os.path.expanduser(manifest_cache_dir)
        )
        os.makedirs(self.manifest_cache_dir)

    @property
    def prompt(self):
        pattern = r"^((.|\n)*)\{([^}]+)\}$"

        # Use re.search to find the match in the example_template string
        match = re.search(pattern, self._prompt.value)
        return match.group(1)

    def _get_ipython_height(self):
        return "600px"

    def _build_component(self, df: "DataFrame") -> "Component":
        from manifest import Manifest

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
                    "classes": "font-mono font-bold bg-slate-200 rounded-md px-2 text-violet-600 w-fit",  # noqa: E501
                }

        @mk.endpoint
        def on_edit(
            df: mk.DataFrame, column: str, keyidx: any, posidx: int, value: any
        ):
            df.loc[keyidx, column] = value

        @mk.reactive(nested_return=True)
        def update_prompt(
            df: mk.DataFrame, template: mk.Store[str], instruction: mk.Store[str]
        ):
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

            train_df = df[df["is_train"]]
            in_context_examples = "\n".join(train_df["example"]())
            prompt = f"{instruction} {in_context_examples} {template}"
            return df, prompt

        @mk.reactive()
        def check_example_template(example_template: str, df: mk.DataFrame):
            example_template = example_template.strip()
            # check if example_template ends with {.*} using a
            # regex and extract the content between the brackets.
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
            instruct_cmd: str,
            df: mk.DataFrame,
            output_col: str,
            selected: list,
            api: str,
        ):
            client_name, engine = api.split("/")
            manifest = Manifest(
                client_name=client_name,
                client_connection=os.getenv("OPENAI_API_KEY"),
                engine=engine,
                temperature=0,
                max_tokens=1,
                cache_name="sqlite",
                cache_connection=os.path.join(self.manifest_cache_dir, "cache.sqlite"),
            )

            def _run_manifest(example: mk.Column):
                # Format instruct-example-instruct prompt.
                out = manifest.run(
                    [f"{instruct_cmd} {in_context_examples} {x}" for x in example]
                )
                return out

            selected_idxs = df.primary_key.isin(selected).to_pandas()

            # Concat all of the in-context examples.
            train_df = df[(~selected_idxs) & (df["is_train"])]
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

        df = df.mark()

        instruction_editor = mk.gui.Editor(
            code="",
            title="Instruction Editor",  # Is this paper theoretical or empirical?",
        )
        example_template_editor = mk.gui.Editor(
            code="",  # "Abstract: {abstract}, Title: {title}; Answer: {answer}",
            title="Training Template Editor",
        )

        output_col = check_example_template(
            example_template=example_template_editor.code, df=df
        )

        df_view, prompt = update_prompt(
            df=df,
            template=example_template_editor.code,
            instruction=instruction_editor.code,
        )

        table = mk.gui.Table(
            df_view, on_edit=on_edit.partial(df=df), classes="h-[400px]"
        )

        api_select = mk.gui.core.Select(
            values=[
                "together/gpt-j-6b",
                "together/gpt-2",
                "openai/text-davinci-003",
                "openai/code-davinci-002",
            ],
            value="together/gpt-j-6b",
        )

        run_manifest_button = mk.gui.Button(
            title="Flash Fill",
            icon="Magic",
            on_click=run_manifest.partial(
                instruct_cmd=instruction_editor.code,
                df=df_view,
                output_col=output_col,
                selected=table.selected,
                api=api_select.value,
            ),
        )
        run_fn = run_manifest.partial(  # noqa: F841
            instruct_cmd=instruction_editor.code,
            output_col=output_col,
            selected=table.selected,
            api=api_select.value,
        )

        formatted_output_col = format_output_col(output_col)

        overview_panel = mk.gui.html.flexcol(
            [
                mk.gui.Text(
                    "Infer selected rows using in-context learning.",
                    classes="font-bold text-slate-600 text-sm",
                ),
                mk.gui.Text(
                    "Specify the instruction and a template for "
                    "in-context train examples.",
                    classes="text-slate-600 text-sm",
                ),
                mk.gui.html.div(
                    [
                        mk.gui.Text(
                            "Target column: ", classes="text-slate-600 text-sm"
                        ),
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
        return (
            mk.gui.html.div(
                [
                    mk.gui.html.grid(
                        [overview_panel, prompt_editor],
                        classes="grid grid-cols-[1fr_3fr] space-x-5",
                    ),
                    mk.gui.html.div([table], classes="h-full w-screen"),
                ],
                classes="gap-4 h-screen grid grid-rows-[auto_1fr]",
            ),
            prompt,
        )
