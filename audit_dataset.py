from pathlib import Path
import pandas as pd
import json

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
REPORT_DIR = PROJECT_DIR / "preprocessing_reports"

CHUNK_SIZE = 50_000

REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

csv_files = sorted(DATASET_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found inside: {DATASET_DIR}"
    )

print(f"Dataset directory: {DATASET_DIR}")
print(f"CSV files found: {len(csv_files)}")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def is_text_column(column_name):
    """
    Identify columns that potentially contain text.
    """
    name = str(column_name).lower()

    return any(
        keyword in name
        for keyword in [
            "text",
            "content",
            "post",
            "comment",
            "message",
            "body"
        ]
    )


def is_label_column(column_name):
    """
    Identify columns that potentially contain labels.
    """
    name = str(column_name).lower()

    return name in {
        "same",
        "label",
        "target",
        "is_same",
        "same_author"
    }


# --------------------------------------------------
# DATASET AUDIT
# --------------------------------------------------

reports = []

for file_path in csv_files:

    print(f"\nInspecting: {file_path.name}")

    report = {
        "file": str(file_path.relative_to(DATASET_DIR)),
        "status": "OK",
        "rows": 0,
        "columns": "",
        "missing_values": {},
        "label_distributions": {},
        "text_statistics": {},
        "dtypes": {},
        "error": ""
    }

    try:

        # Read one chunk at a time to reduce memory usage.
        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            encoding="utf-8"
        ):

            if report["rows"] == 0:
                report["columns"] = list(chunk.columns)

                report["dtypes"] = {
                    col: str(dtype)
                    for col, dtype in chunk.dtypes.items()
                }

                report["missing_values"] = {
                    col: 0 for col in chunk.columns
                }

            report["rows"] += len(chunk)

            # Count missing values.
            for col in chunk.columns:
                report["missing_values"][col] += int(
                    chunk[col].isna().sum()
                )

            # Count label values.
            for col in chunk.columns:
                if is_label_column(col):

                    counts = chunk[col].value_counts(
                        dropna=False
                    )

                    distribution = report[
                        "label_distributions"
                    ].setdefault(col, {})

                    for label, count in counts.items():

                        key = str(label)

                        distribution[key] = (
                            distribution.get(key, 0)
                            + int(count)
                        )

            # Collect text-length statistics.
            for col in chunk.columns:

                if not is_text_column(col):
                    continue

                values = chunk[col].dropna().astype(str)
                lengths = values.str.len()

                stats = report[
                    "text_statistics"
                ].setdefault(
                    col,
                    {
                        "non_missing_texts": 0,
                        "empty_texts": 0,
                        "min_length": None,
                        "max_length": 0,
                        "total_characters": 0
                    }
                )

                stats["non_missing_texts"] += len(values)

                stats["empty_texts"] += int(
                    values.str.strip().eq("").sum()
                )

                if not lengths.empty:

                    chunk_min = int(lengths.min())
                    chunk_max = int(lengths.max())

                    if stats["min_length"] is None:
                        stats["min_length"] = chunk_min
                    else:
                        stats["min_length"] = min(
                            stats["min_length"],
                            chunk_min
                        )

                    stats["max_length"] = max(
                        stats["max_length"],
                        chunk_max
                    )

                    stats["total_characters"] += int(
                        lengths.sum()
                    )

        # Calculate average text length.
        for stats in report["text_statistics"].values():

            count = stats["non_missing_texts"]

            stats["average_length"] = (
                round(
                    stats["total_characters"] / count,
                    2
                )
                if count > 0
                else 0
            )

        print("Rows:", report["rows"])
        print("Columns:", report["columns"])

    except Exception as exc:

        report["status"] = "ERROR"
        report["error"] = str(exc)

        print("ERROR:", exc)

    reports.append(report)


# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

report_path = REPORT_DIR / "dataset_audit.json"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        reports,
        file,
        indent=4,
        ensure_ascii=False
    )

# Create a simpler summary CSV.
summary = []

for report in reports:

    summary.append({
        "file": report["file"],
        "status": report["status"],
        "rows": report["rows"],
        "columns": ", ".join(
            report["columns"]
            if isinstance(report["columns"], list)
            else []
        ),
        "error": report["error"]
    })

summary_path = REPORT_DIR / "dataset_summary.csv"

pd.DataFrame(summary).to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 50)
print("DATASET AUDIT COMPLETE")
print("=" * 50)

print(f"Detailed report: {report_path}")
print(f"Summary report:  {summary_path}")