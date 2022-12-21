
# Operations

Meerkat provides a set of data-wrangling operations that can be applied over DataFrames. This page provides an overview of the operations available in Meerkat. 


### Mapping
- {func}`~meerkat.map`: Applies a function to each row of a DataFrame and returns a {class}`~meerkat.Column` or {class}`~meerkat.DataFrame` with the resulting values.
- {func}`~meerkat.update`: Same as {func}`~meerkat.map`, but adds the new column to the original DataFrame in-place. 
- {func}`~meerkat.filter`: Applies a function to each row of a DataFrame and returns a new DataFrame with only the rows for which the function returned `True`.

### Combining
- {func}`~meerkat.concat`: Combines multiple DataFrames by stacking them vertically or horizontally.
- {func}`~meerkat.merge`: Joins two DataFrames based on common columns or indices.

### Grouping
- {func}`~meerkat.groupby`: Groups a DataFrame by a specified column(s) and applies a function to each group.
- {func}`~meerkat.clusterby`: Groups a DataFrame by a specified column or index and applies a function to each group, returning a new DataFrame with the resulting values.
- {func}`~meerkat.explainby`: Groups a DataFrame by a specified column or index and returns summary statistics for each group.

### Aggregating
- {func}`~meerkat.aggregate`: Applies an aggregation function to each column of a DataFrame. 

### Sort, Sample
- {func}`~meerkat.sort`: Sorts a DataFrame by one or more columns.
- {func}`~meerkat.sample`: Returns a new DataFrame with a random sample of rows.