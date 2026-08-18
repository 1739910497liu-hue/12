import argparse
import csv
import json
import re
from pathlib import Path


CORE_FIELDS = [
    ("voltage_v", "Voltage", "V", 30.0),
    ("current_a", "current", "A", 300.0),
    ("speed_cm_min", "speed", "cm/min", 50.0),
    ("gas_flow_l_min", "gas flow", "L/min", 50.0),
    ("plate_thickness_mm", "plate thickness", "mm", 20.0),
]


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_text(value):
    return re.sub(r"\s+", " ", str(value).replace("\\", "/")).strip()


def first_match(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return as_float(match.group(1))
    return None


def extract_from_path(path_text):
    text = clean_text(path_text)
    return {
        "voltage_v": first_match([
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*V(?![A-Za-z])",
        ], text),
        "current_a": first_match([
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*A(?![A-Za-z])",
        ], text),
        "speed_cm_min": first_match([
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
    }


def extract_from_text(prompt_text):
    text = clean_text(prompt_text)
    return {
        "voltage_v": first_match([
            r"(?:arc\s*|welding\s*)?voltage[^.\d]{0,40}(\d+(?:\.\d+)?)\s*V\b",
            r"(\d+(?:\.\d+)?)\s*V[^.]{0,40}(?:arc\s*|welding\s*)?voltage\b",
            r"parameters[^.]{0,120}(\d+(?:\.\d+)?)\s*V\b",
        ], text),
        "current_a": first_match([
            r"(?:welding\s*)?current[^.\d]{0,40}(\d+(?:\.\d+)?)\s*A\b",
            r"(\d+(?:\.\d+)?)\s*A[^.]{0,40}(?:welding\s*)?current\b",
            r"parameters[^.]{0,120}(\d+(?:\.\d+)?)\s*A\b",
        ], text),
        "speed_cm_min": first_match([
            r"(?:welding\s*)?speed[^.\d]{0,40}(\d+(?:\.\d+)?)\s*cm\s*/?\s*min\b",
            r"(\d+(?:\.\d+)?)\s*cm\s*/?\s*min[^.]{0,40}(?:welding\s*)?speed\b",
        ], text),
        "gas_flow_l_min": first_match([
            r"gas\s*flow(?:\s*rate)?[^.\d]{0,40}(\d+(?:\.\d+)?)\s*L\s*/?\s*min\b",
            r"flow\s*rate[^.\d]{0,40}(\d+(?:\.\d+)?)\s*L\s*/?\s*min\b",
            r"(\d+(?:\.\d+)?)\s*L\s*/?\s*min[^.]{0,60}(?:gas\s*flow|argon|shielding)",
            r"argon\s*shielding\s*gas\s*\((\d+(?:\.\d+)?)\s*L\s*/?\s*min\)",
        ], text),
        "plate_thickness_mm": first_match([
            r"(?:SS304|material|base\s*material)[^.]{0,60}(\d+(?:\.\d+)?)\s*mm\s*plate\b",
            r"(\d+(?:\.\d+)?)\s*mm\s*plate\b",
        ], text),
    }


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


def choose_values(path_values, text_values):
    values = {}
    sources = {}
    for field, _, _, _ in CORE_FIELDS:
        if path_values.get(field) is not None:
            values[field] = path_values[field]
            sources[field] = "path"
        elif text_values.get(field) is not None:
            values[field] = text_values[field]
            sources[field] = "text"
        else:
            values[field] = None
            sources[field] = None
    return values, sources


def format_parameter_sentence(values):
    parts = []
    for field, label, unit, _ in CORE_FIELDS:
        value = values.get(field)
        if value is None:
            parts.append(f"{label} unknown")
        else:
            parts.append(f"{label} {value:g}{unit}")
    return "Parameters: " + ", ".join(parts) + "."


def make_numeric_vector(values):
    normalized = []
    present = []
    for field, _, _, scale in CORE_FIELDS:
        value = values.get(field)
        if value is None:
            normalized.append(0.0)
            present.append(0.0)
        else:
            normalized.append(float(value) / scale)
            present.append(1.0)
    return normalized + present


def main():
    parser = argparse.ArgumentParser(description="Extract Voltage/current/speed/gas-flow parameters from welding JSON.")
    parser.add_argument("--input", default=r"H:\数据集\JSON\text\train_text.json")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    out_json = Path(args.out_json) if args.out_json else input_path.with_name(input_path.stem + "_parameters.json")
    out_csv = Path(args.out_csv) if args.out_csv else input_path.with_name(input_path.stem + "_parameters.csv")

    items = load_items(input_path)
    output = {
        "_meta": {
            "source": str(input_path),
            "count": len(items),
            "fields": [field for field, _, _, _ in CORE_FIELDS],
            "parameter_text_format": "Parameters: Voltage 17.2V, current 200A, speed 19cm/min, gas flow 30L/min, plate thickness 5mm.",
            "numeric_vector": "normalized values followed by presence flags in field order",
            "selection_rule": "path value first, then prompt text value",
        },
        "items": {},
    }
    coverage = {field: 0 for field, _, _, _ in CORE_FIELDS}
    rows = []

    for key, prompt in items:
        path_values = extract_from_path(key)
        text_values = extract_from_text(prompt)
        values, sources = choose_values(path_values, text_values)
        parameter_text = format_parameter_sentence(values)
        numeric_vector = make_numeric_vector(values)
        for field in coverage:
            if values[field] is not None:
                coverage[field] += 1

        output["items"][key] = {
            "parameter_text": parameter_text,
            "values": values,
            "value_sources": sources,
            "numeric_vector": numeric_vector,
            "original_text": prompt,
        }
        row = {
            "image_key": key,
            "parameter_text": parameter_text,
            "voltage_v": values["voltage_v"],
            "current_a": values["current_a"],
            "speed_cm_min": values["speed_cm_min"],
            "gas_flow_l_min": values["gas_flow_l_min"],
            "plate_thickness_mm": values["plate_thickness_mm"],
            "voltage_source": sources["voltage_v"],
            "current_source": sources["current_a"],
            "speed_source": sources["speed_cm_min"],
            "gas_flow_source": sources["gas_flow_l_min"],
            "plate_thickness_source": sources["plate_thickness_mm"],
            "vec_0_voltage": numeric_vector[0],
            "vec_1_current": numeric_vector[1],
            "vec_2_speed": numeric_vector[2],
            "vec_3_gas_flow": numeric_vector[3],
            "vec_4_plate_thickness": numeric_vector[4],
            "vec_5_voltage_present": numeric_vector[5],
            "vec_6_current_present": numeric_vector[6],
            "vec_7_speed_present": numeric_vector[7],
            "vec_8_gas_flow_present": numeric_vector[8],
            "vec_9_plate_thickness_present": numeric_vector[9],
            "original_text": prompt,
        }
        rows.append(row)

    output["_meta"]["coverage"] = coverage

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_key"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input records: {len(items)}")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote CSV: {out_csv}")
    for field in coverage:
        print(f"{field}: {coverage[field]}/{len(items)}")


if __name__ == "__main__":
    main()
