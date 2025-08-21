from pydantic import BaseModel


def get_field(model: BaseModel, field: str):
    try:
        fields = field.split(".")
        fields[0]
        for field in fields:
            model = getattr(model, field)
        return model
    except AttributeError as e:
        raise e
