import argparse
import yaml
from pydantic import ValidationError

from dataqa.core.data_models.asset_models import DatabaseSchema, Examples, Rules

def validate_yaml(file_path: str, model, label) -> None:
    print(f"\nValidating {label} ({file_path})...")
    try:
        with open(file_path, mode="r") as f:
            data = yaml.safe_load(stream=f)
        model.validate(data)
        print(f"{label} is valid!")
    except FileNotFoundError:
        print(f"{label} file not found: {file_path}")
    except ValidationError as e:
        print(f"Validation errors in {label}:")
        print(e.json())
    except Exception as e:
        print(f"Unexpected error while validating {label}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate schema, rules, and examples YAML files."
    )
    parser.add_argument("-s", "--schema", type=str, help="Path to schema.yaml")
    parser.add_argument("-r", "--rules", type=str, help="Path to rules.yaml")
    parser.add_argument(
        "-e", "--examples", type=str, help="Path to examples.yaml"
    )

    args = parser.parse_args()

    if not (args.schema or args.rules or args.examples):
        print(
            "No files provided. Use --schema, --rules, or --examples to specify files to validate."
        )
        return

    if args.schema:
        validate_yaml(file_path=args.schema, model=DatabaseSchema, label="schema.yaml")
    if args.rules:
        validate_yaml(file_path=args.rules, model=Rules, label="rules.yaml")
    if args.examples:
        validate_yaml(file_path=args.examples, model=Examples, label="examples.yaml")


if __name__ == "__main__":
    main()