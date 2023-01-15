import meerkat as mk
import datetime

df = mk.DataFrame(
    {
        "birth_year": [1967, 1993, 2010, 1985, 2007, 1990, 1943],
        "residence": ["MA", "LA", "NY", "NY", "MA", "MA", "LA"],
    }
)
df["age"] = df["birth_year"].map(
    lambda x: datetime.datetime.now().year - x
)

def is_eligibile(age, residence):
    old_enough = age >= 18
    return (residence == "MA") and old_enough, (residence == "LA") and old_enough

df.map(is_eligibile)