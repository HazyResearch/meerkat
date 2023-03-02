# TODO(karan): fix and generalize these APIs
import functools
import random
from typing import List

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from meerkat.state import state

router = APIRouter(
    prefix="/llm",
    tags=["llm"],
    responses={404: {"description": "Not found"}},
)

EmbeddedBody = functools.partial(Body, embed=True)


class CategoryGenerationResponse(BaseModel):
    categories: List[str]


class CategorizationGenerationResponse(BaseModel):
    categories: List[str]


@router.post("/generate/categories")
def generate_categories(
    dataset_description: str = EmbeddedBody(),
    hint: str = EmbeddedBody(),
    n_categories: int = EmbeddedBody(default=20),
) -> CategoryGenerationResponse:
    """Generate a list of categories for a dataset using an LLM."""

    from manifest import Prompt

    state.llm.set(client="ai21", engine="j1-jumbo")

    try:
        response = state.llm.get().run(
            Prompt(
                f"""
List {n_categories} distinct attributes that can be used to tag a dataset.
---
Dataset: {dataset_description}
Hint: {hint}
Attributes:
"""
            ),
            top_k_return=1,
            temperature=0.5,
            max_tokens=128,
            stop_token="\n---",
        )
        print(response)
        lines = response.split("\n")
        categories = [line.split(". ")[-1] for line in lines]

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CategoryGenerationResponse(
        categories=categories,
    )


@router.post("/generate/categorization")
def generate_categorization(
    description: str = EmbeddedBody(),
    existing_categories: List[str] = EmbeddedBody(default=["none"]),
) -> CategorizationGenerationResponse:
    """Generate a list of categories for a dataset using an LLM."""
    from manifest import Prompt

    state.llm.set(client="ai21", engine="j1-jumbo")

    random.shuffle(existing_categories)

    try:
        response = state.llm.get().run(
            Prompt(
                f"""
List 5 distinct possibilities that could be used to categorize an attribute. The attribute description is provided.
---
Attribute description: types of eyewear
Existing categories: none
Additional categories:
1. sunglasses
2. eyeglasses
3. goggles
4. blindfolds
5. vr headsets
---
Attribute description: age groups (words not ranges)
Existing categories: infants, teenagers
Additional categories:
1. toddlers
2. children
3. teens
4. adults
5. seniors
---
Attribute description: styles of art
Existing categories: abstract, modern, impressionism
Additional categories:
1. realism
2. surrealism
3. cubism
4. psychedelic
5. contemporary
---
Attribute description: political leanings
Existing categories: none
Additional categories:
1. authoritarian
2. libertarian
3. liberal
4. conservative
5. anarchist
---
Attribute description: {description}
Existing categories: {", ".join(existing_categories)}
More categories:
"""  # noqa: E501
            ),
            top_k_return=1,
            temperature=0.5,
            max_tokens=128,
            stop_token="\n---",
        )
        print(response)
        lines = response.split("\n")
        categories = [line.split(". ")[-1] for line in lines]

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CategorizationGenerationResponse(
        categories=categories,
    )
