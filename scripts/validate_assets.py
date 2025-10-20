import argparse
import yaml
from pydantic import ValidationError

from dataqa.core.data_models.asset_models import DatabaseSchema, Examples, Rules

def validate_yaml(file_path, model, label):
    print(f"\nValidating {label} ({file_path})...")
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        model(**data)
        print(f"{label} is valid!")
    except FileNotFoundError:
        print(f"{label} file not found: {file_path}")
    except ValidationError as e:
        print(f"Validation errors in {label}:")
        print(e.json())
    except Exception as e:
        print(f"Unexpected error while validating {label}: {e}")


def main():
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
        validate_yaml(args.schema, DatabaseSchema, "schema.yaml")
    if args.rules:
        validate_yaml(args.rules, Rules, "rules.yaml")
    if args.examples:
        validate_yaml(args.examples, Examples, "examples.yaml")


if __name__ == "__main__":
    main()
