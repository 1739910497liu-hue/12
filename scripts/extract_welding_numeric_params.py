import argparse
import csv
import json
import re
from pathlib import Path


FIELD_SPECS = [
    ("current_a", 300.0),
    ("voltage_v", 30.0),
    ("welding_speed_cm_min", 50.0),
    ("gas_flow_l_min", 50.0),
    ("plate_thickness_mm", 20.0),
    ("lens_mm", 100.0),
    ("focal_length_mm", 100.0),
    ("speed_level", 30.0),
]


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_match(patterns, text, group=1):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return as_float(match.group(group))
    return None


def clean_text(value):
    return re.sub(r"\s+", " ", str(value).replace("\\", "/")).strip()


def extract_from_path(path_text):
    text = clean_text(path_text)
    return {
        "current_a": first_match([
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*A(?![A-Za-z])",
        ], text),
        "voltage_v": first_match([
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*V(?![A-Za-z])",
        ], text),
        "welding_speed_cm_min": first_match([
            r"(?:welding\s*speed|w\.s\.)\s*(\d+(?:\.\d+)?)\s*cm(?:\.m|/min|min)?",
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*cm(?:\.m|/min|min)\b",
        ], text),
        "gas_flow_l_min": first_match([
            r"g\.?\s*f\.?\s*(\d+(?:\.\d+)?)\s*L(?:\.m|/min|min)?",
            r"gas\s*flow(?:\s*rate)?\s*(\d+(?:\.\d+)?)\s*L(?:\.m|/min|min)?",
        ], text),
        "plate_thickness_mm": first_match([
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*mm\s*Plate",
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*mmPlate",
        ], text),
        "lens_mm": first_match([
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*mm\s*Lens",
        ], text),
        "focal_length_mm": None,
        "speed_level": first_match([
            r"(?:Lev|level)\s*(\d+(?:\.\d+)?)",
        ], text),
    }


def extract_from_prompt(prompt_text):
    text = clean_text(prompt_text)
    return {
        "current_a": first_match([
            r"(?:welding\s*)?current[^.\d]{0,40}(\d+(?:\.\d+)?)\s*A\b",
            r"(\d+(?:\.\d+)?)\s*A[^.]{0,40}(?:welding\s*)?current\b",
            r"parameters[^.]{0,80}(\d+(?:\.\d+)?)\s*A\b",
        ], text),
        "voltage_v": first_match([
            r"(?:arc\s*|welding\s*)?voltage[^.\d]{0,40}(\d+(?:\.\d+)?)\s*V\b",
            r"(\d+(?:\.\d+)?)\s*V[^.]{0,40}(?:arc\s*|welding\s*)?voltage\b",
        ], text),
        "welding_speed_cm_min": first_match([
            r"(?:welding\s*)?speed[^.\d]{0,40}(\d+(?:\.\d+)?)\s*cm\s*/?\s*min\b",
            r"(\d+(?:\.\d+)?)\s*cm\s*/?\s*min[^.]{0,40}(?:welding\s*)?speed\b",
        ], text),
        "gas_flow_l_min": first_match([
            r"gas\s*flow(?:\s*rate)?[^.\d]{0,40}(\d+(?:\.\d+)?)\s*L\s*/?\s*min\b",
            r"flow\s*rate[^.\d]{0,40}(\d+(?:\.\d+)?)\s*L\s*/?\s*min\b",
            r"(\d+(?:\.\d+)?)\s*L\s*/?\s*min[^.]{0,50}gas\s*flow\b",
            r"argon\s*shielding\s*gas\s*\((\d+(?:\.\d+)?)\s*L\s*/?\s*min\)",
        ], text),
        "plate_thickness_mm": first_match([
            r"(?:SS304|material|base\s*material)[^.]{0,50}(\d+(?:\.\d+)?)\s*mm\s*plate\b",
            r"(\d+(?:\.\d+)?)\s*mm\s*plate\b",
        ], text),
        "lens_mm": first_match([
            r"lens[^.\d]{0,30}(\d+(?:\.\d+)?)\s*mm\b",
            r"(\d+(?:\.\d+)?)\s*mm\s*lens\b",
        ], text),
        "focal_length_mm": first_match([
            r"focal\s*length[^.\d]{0,30}(\d+(?:\.\d+)?)\s*mm\b",
        ], text),
        "speed_level": first_match([
            r"(?:speed\s*)?level[^.\d]{0,10}(\d+(?:\.\d+)?)",
            r"\bLev\s*(\d+(?:\.\d+)?)\b",
        ], text),
    }


def choose_values(path_values, prompt_values):
    selected = {}
    sources = {}
    for field, _ in FIELD_SPECS:
        if path_values.get(field) is not None:
            selected[field] = path_values[field]
            sources[field] = "path"
        elif prompt_values.get(field) is not None:
            selected[field] = prompt_values[field]
            sources[field] = "text"
        else:
            selected[field] = None
            sources[field] = None
    return selected, sources


def make_vectors(values):
    raw = []
    normalized = []
    present = []
    for field, scale in FIELD_SPECS:
        value = values.get(field)
        if value is None:
            raw.append(None)
            normalized.append(0.0)
            present.append(0.0)
        else:
            raw.append(float(value))
            normalized.append(float(value) / scale)
            present.append(1.0)
    return raw, normalized + present


def load_items(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.items())
    if isinstance(data, list):
        items = []
        for item in data:
            if not isinstance(item, dict):
                continue
            key = item.get("file") or item.get("path") or item.get("image") or item.get("image_path")
            text = item.get("caption") or item.get("prompt") or item.get("text")
            if key is not None and text is not None:
                items.append((key, text))
        return items
    raise ValueError("Input JSON must be a dict or a list of records.")


def main():
    parser = argparse.ArgumentParser(description="Extract numeric welding parameters from train_text.json.")
    parser.add_argument("--input", default=r"H:\数据集\JSON\text\train_text.json")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    out_json = Path(args.out_json) if args.out_json else input_path.with_name(input_path.stem + "_numeric.json")
    out_csv = Path(args.out_csv) if args.out_csv else input_path.with_name(input_path.stem + "_numeric.csv")

    items = load_items(input_path)
    output = {
        "_meta": {
            "source": str(input_path),
            "count": len(items),
            "fields": [field for field, _ in FIELD_SPECS],
            "numeric_vector": "normalized values followed by presence flags in the same field order",
            "selection_rule": "path value first, then prompt text value",
        },
        "items": {},
    }

    coverage = {field: 0 for field, _ in FIELD_SPECS}
    rows = []
    for key, prompt in items:
        path_values = extract_from_path(key)
        prompt_values = extract_from_prompt(prompt)
        values, sources = choose_values(path_values, prompt_values)
        raw_vector, numeric_vector = make_vectors(values)
        for field, _ in FIELD_SPECS:
            if values[field] is not None:
                coverage[field] += 1
        record = {
            "text": prompt,
            "values": values,
            "value_sources": sources,
            "path_values": path_values,
            "text_values": prompt_values,
            "raw_vector": raw_vector,
            "numeric_vector": numeric_vector,
        }
        output["items"][key] = record
        row = {"image_key": key, "text": prompt}
        row.update(values)
        row.update({f"{field}_source": sources[field] for field, _ in FIELD_SPECS})
        row.update({f"raw_{idx}_{field}": raw_vector[idx] for idx, (field, _) in enumerate(FIELD_SPECS)})
        row.update({f"vec_{idx}": numeric_vector[idx] for idx in range(len(numeric_vector))})
        rows.append(row)

    output["_meta"]["coverage"] = coverage

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["image_key", "text"]
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input records: {len(items)}")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote CSV: {out_csv}")
    for field, _ in FIELD_SPECS:
        print(f"{field}: {coverage[field]}/{len(items)}")


if __name__ == "__main__":
    main()
