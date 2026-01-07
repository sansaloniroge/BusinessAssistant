from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


Operator = Literal["$eq", "$in", "$contains_any", "$gte", "$lte"]


class FieldFilter(BaseModel):
    op: Operator
    value: Any


MetaFilters = dict[str, str | int | float | bool | FieldFilter]
