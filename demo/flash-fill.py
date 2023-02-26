"""Flash-fill"""
from functools import partial

from manifest import Manifest
import re

import meerkat as mk
from meerkat.dataframe import Batch

manifest = Manifest(
    client_name="huggingface",
    client_connection="http://127.0.0.1:7861",
)


def complete_prompt(row, example_template: mk.Store[str]):
    assert isinstance(row, dict)
    output = example_template.format(**row)
    return output


filepath = "/Users/sabrieyuboglu/Downloads/arxiv-metadata-oai-snapshot.json"
df = mk.from_json(filepath=filepath, lines=True, backend="arrow")
df["url"] = "https://arxiv.org/pdf/" + df["id"]
df["pdf"] = mk.files(
    df["url"], cache_dir="/Users/sabrieyuboglu/Downloads/pdf-cache", type="pdf"
)

df = df[df["categories"].str.contains("stat.ML")]


df = df[
    ["id", "authors", "title", "journal-ref", "categories", "abstract", "pdf", "url"]
]
df["note"] = mk.scalar([""] * len(df), backend="arrow")



df = df.mark()


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
            raise ValueError(f"The column '{content}' does not exist in the dataframe.")
    else:
        raise ValueError("The example template must end with '{" + content + "}'")
    return content


@mk.endpoint()
def run_manifest(instruct_cmd: str, df: mk.DataFrame, output_col: str, selected: list):
    def _run_manifest(example: Batch):
        # Format instruct-example-instruct prompt.
        return ["test" for x in example]
        print("before")
        out = manifest.run(
            [f"{instruct_cmd} {in_context_examples} {x}" for x in example]
        )
        print("after")
        return out
    selected_idxs = df.primary_key.isin(selected)

    # Concat all of the in-context examples.
    train_df = df[~selected_idxs & df[output_col] != ""]
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


@mk.reactive()
def update_df_with_example_template(df: mk.DataFrame, template: mk.Store[str]):
    """Update the df with the new prompt template.
    This is as simple as returning a view.
    """
    df = df.view()

    # Extract out the example template from the prompt template.
    df["example"] = mk.defer(
        df, function=partial(complete_prompt, example_template=example_template)
    )
    return df


@mk.endpoint()
def set_code(code: mk.Store[str], new_code: str):
    code.set(new_code)


instruction_editor = mk.gui.Editor(
    code="Write a note for my guest.", title="Instruction Editor"
)
example_template_editor = mk.gui.Editor(
    code="Abstract: {abstract}, Title: {title}; Note: {note}",
    title="Training Template Editor",
)

example_template = example_template_editor.code

output_col = check_example_template(example_template=example_template, df=df)

df_view = update_df_with_example_template(df, example_template)


@mk.endpoint
def on_edit(df: mk.DataFrame, column: str, keyidx: any, posidx: int, value: any):
    df.loc[keyidx, column] = value


# mk.gui.Gallery(df_view, main_column="guest")
table = mk.gui.Table(
    df_view[["id", "authors", "title", "abstract", "categories", "pdf", "note", "example"]],
    on_edit=on_edit.partial(df=df),
)

mk.gui.html.button()
run_manifest_button = mk.gui.Button(
    title="Flash Fill",
    icon="Magic",
    on_click=run_manifest.partial(
        instruct_cmd=instruction_editor.code,
        df=df_view,
        output_col=output_col,
        selected=table.selected,
    ),
)


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
        mk.gui.html.flex(
            [
                mk.gui.Text("Target column: ", classes="text-slate-600 text-sm"),
                mk.gui.Text(
                    output_col,
                    classes="font-mono text-violet-600 font-bold bg-slate-200 rounded-md px-2",
                ),
            ],
            classes="gap-3 align-middle",
        ),
        run_manifest_button,
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


page = mk.gui.Page(
    component=mk.gui.html.div(
        [
            mk.gui.html.grid(
                [overview_panel, prompt_editor],
                classes="grid grid-cols-[1fr_3fr] space-x-5",
            ),
            mk.gui.html.div([table], classes="h-full w-screen"),
        ],
        classes="gap-4 h-screen grid grid-rows-[auto_1fr]",
    ),
    id="flash-fill",
    progress=False,
)

page.launch()
