from typing import List, Optional, Type

from pydantic import BaseModel, Field, create_model

from dataqa.core.components.base_component import Variable


def build_base_model_from_parameters(
    base_model_name: str, parameters: List[Variable]
) -> Type[BaseModel]:
    """
    Dynamically build `base_model_name` as a Pydantic BaseModel class.
    The new class contains all the variable in parameters as fields.
    """
    model_fields = {}
    for field_properties in parameters:
        field_name = field_properties.name
        field_type = eval(
            field_properties.type
        )  # TODO if we can avoid using `eval`
        field_description = field_properties.description
        default = field_properties.default
        optional = field_properties.optional
        if optional:
            field_type = Optional[field_type]
            model_fields[field_name] = (
                field_type,
                Field(description=field_description, default=default),
            )
        else:
            model_fields[field_name] = (
                field_type,
                Field(..., description=field_description),
            )

    return create_model(base_model_name, **model_fields)


def extract(
    response: str, prefix: str, suffix: str, error_tolerant: bool = True
) -> str:
    """
    Parse the response and return the text between the first `prefix` and the last `suffix`.
    """
    if len(prefix) == 0:
        a = 0
    else:
        a = response.find(prefix)
    b = response.rfind(suffix)
    if a < 0 or b < 0:
        if error_tolerant:
            return ""
        raise ValueError(
            f"can not find keywords {prefix} or {suffix} in {response}"
        )
    return response[a + len(prefix) : b].strip()
