import meerkat as mk

# Create a DataFrame
df = mk.get("imagenette")

# List of unique labels represented in the DataFrame
labels = list(df['label'].unique())

# Display the labels as choices to select from
choices = mk.gui.Choice(choices=labels, value=labels[0], title="Choose a Label")

# Create a reactive function that returns a filtered view of the DataFrame
# @mk.gui.reactive
@mk.gui.react()
def filter_by_label(df: mk.DataFrame, label: str):
    return df[df['label'] == label]

# with mk.gui.react():
# Reactively run filter by label, on top of df and choices.value
df_filtered = filter_by_label(df, choices.value)
    
# Visualize the filtered_df
df_viz = mk.gui.Table(df=df_filtered)

interface = mk.gui.Interface(
    component=mk.gui.RowLayout(components=[choices, df_viz]),
)
interface.write_component_wrappers()
interface.write_sveltekit_route()
breakpoint()

mk.gui.start()
interface.launch()
