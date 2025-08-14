from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter


def repr_str(dumper: RoundTripRepresenter, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


out_yaml = YAML()
out_yaml.representer.add_representer(str, repr_str)
