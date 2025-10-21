from typing import List, Optional, Type

from pydantic import BaseModel, Field, create_model


def create_base_model(
    model_name: str,
    parameters: List,
    parent_model: Optional[Type[BaseModel]] = None,
) -> BaseModel:
    """
    Create Pydantic base model dynamically
    :param model_name: name of the base model to be created
    :param parameters: list of fields as dictionary
    :param parent_model: class of parent base model
    :return: created base model
    """
    model_fields = {}
    for field in parameters:
        field_name = field["name"]
        field_type = eval(field["type"])
        field_description = field["description"]
        model_fields[field_name] = (
            field_type,
            Field(description=field_description),
        )
    if parent_model is None:
        return create_model(model_name, **model_fields)
    else:
        return create_model(model_name, __base__=parent_model, **model_fields)
