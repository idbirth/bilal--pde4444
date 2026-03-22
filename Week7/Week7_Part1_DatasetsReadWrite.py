#!/usr/bin/env python3
"""Week 7 Part 1: Dataset read/write examples converted from notebook."""

from pathlib import Path
import csv
import json


BASE_DIR = Path(__file__).resolve().parent
TEXT_FILE = BASE_DIR / "accounts.txt"
JSON_FILE = BASE_DIR / "accounts.json"
CSV_FILE = BASE_DIR / "accounts.csv"


def write_text_file() -> None:
    """Write sample account rows to a text file."""
    with TEXT_FILE.open("w", encoding="utf-8") as accounts:
        accounts.write("100 Jones 24.98\n")
        accounts.write("200 Doe 345.67\n")
        accounts.write("300 White 0.00\n")
        accounts.write("400 Stone -42.16\n")
        accounts.write("500 Rich 224.62\n")
        print("600 Jones 24.98", file=accounts)


def read_text_file() -> None:
    """Read and print the text file in table format."""
    print("\nText File Contents")
    print(f'{"Account":<10}{"Name":<10}{"Balance":>10}')
    with TEXT_FILE.open("r", encoding="utf-8") as accounts:
        for record in accounts:
            account, name, balance = record.split()
            print(f"{account:<10}{name:<10}{balance:>10}")


def write_json_file() -> None:
    """Write sample account rows to a JSON file."""
    accounts_dict = {
        "accounts": [
            {"account": 100, "name": "Jones", "balance": 24.98},
            {"account": 200, "name": "Doe", "balance": 345.67},
        ]
    }
    with JSON_FILE.open("w", encoding="utf-8") as accounts:
        json.dump(accounts_dict, accounts)


def read_json_file() -> None:
    """Read JSON and print object-level examples from the notebook."""
    with JSON_FILE.open("r", encoding="utf-8") as accounts:
        accounts_json = json.load(accounts)

    print("\nJSON full object:")
    print(accounts_json)
    print("\nJSON rows:")
    print(accounts_json["accounts"])
    print("\nJSON first row:")
    print(accounts_json["accounts"][0])
    print("\nJSON second row:")
    print(accounts_json["accounts"][1])


def pretty_print_json() -> None:
    """Pretty print JSON using dumps(..., indent=4)."""
    print("\nPretty JSON (dumps):")
    with JSON_FILE.open("r", encoding="utf-8") as accounts:
        print(json.dumps(json.load(accounts), indent=4))


def write_csv_file() -> None:
    """Write sample account rows to a CSV file."""
    with CSV_FILE.open("w", newline="", encoding="utf-8") as accounts:
        writer = csv.writer(accounts)
        writer.writerow([100, "Jones", 24.98])
        writer.writerow([200, "Doe", 345.67])
        writer.writerow([300, "White", 0.00])
        writer.writerow([400, "Stone", -42.16])
        writer.writerow([500, "Rich", 224.62])


def read_csv_file() -> None:
    """Read and print CSV file in table format."""
    print("\nCSV File Contents")
    print(f'{"Account":<10}{"Name":<10}{"Balance":>10}')
    with CSV_FILE.open("r", newline="", encoding="utf-8") as accounts:
        reader = csv.reader(accounts)
        for record in reader:
            account, name, balance = record
            print(f"{account:<10}{name:<10}{balance:>10}")


def main() -> None:
    write_text_file()
    read_text_file()

    write_json_file()
    read_json_file()
    pretty_print_json()

    write_csv_file()
    read_csv_file()


if __name__ == "__main__":
    main()
