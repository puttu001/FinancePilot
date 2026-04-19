from pydantic import BaseModel
from typing import List, Annotated

class IngestionInput(BaseModel):
    file: Annotated[]
