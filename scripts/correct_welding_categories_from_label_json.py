import argparse
import json
import re
from pathlib import Path


LABEL_TO_DEFECT = {
    0: "burn through",
    1: "contamination",
    2: "good weld",
    3: "high travel speed",
    4: "lack of fusion",
    5: "lack of shielding gas",
}


def clean_text(value):
    return re.sub(r"\s+", " ", str(value).replace("\\", "/")).strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_welding_type(path_text, prompt_text):
    text = f"{path_text} {prompt_text}".lower()
    if re.search(r"\btig\b|tungsten\s+inert\s+gas", text):
        return "TIG"
    if re.search(r"\bmig\b|metal\s+inert\s+gas", text):
        return "MIG"
    if re.search(r"\bmag\b|metal\s+active\s+gas", text):
        return "MAG"
    if re.search(r"laser\s+weld", text):
        return "laser welding"
    if re.search(r"arc\s+weld", text):
        return "arc welding"
    return None


def find_material(path_text, prompt_text):
    text = f"{path_text} {prompt_text}"
    material_patterns = [
        (r"\bSS\s*304\b|\bSS304\b|304\s*stainless\s*steel", "SS304"),
        (r"\b316L\b|SS\s*316L|SS316L", "SS316L"),
        (r"\b304L\b|SS\s*304L|SS304L", "SS304L"),
        (r"stainless\s*steel", "stainless steel"),
        (r"aluminum|aluminium", "aluminum"),
        (r"carbon\s*steel", "carbon steel"),
    ]
    for pattern, label in material_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return None


def make_category_text(welding_type, material, defect_type):
    return (
        "Categories: "
        f"welding type {welding_type or 'unknown'}, "
        f"material {material or 'unknown'}, "
        f"defect type {defect_type or 'unknown'}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Correct welding categories with an explicit numeric label JSON."
    )
    parser.add_argument("--text-json", default=r"H:\数据集\JSON\text\train_text.json")
    parser.add_argument("--label-json", default=r"H:\数据集\JSON\train - 副本.json")
    parser.add_argument("--out-json", default=r"H:\数据集\JSON\text\train_text_categories_label_corrected.json")
    parser.add_argument(
        "--out-original-format-json",
        default=r"H:\数据集\JSON\text\train_text_categories_label_corrected_original_format.json",
    )
    args = parser.parse_args()

    text_json = Path(args.text_json)
    label_json = Path(args.label_json)
    if not text_json.exists():
        raise FileNotFoundError(f"Text JSON not found: {text_json}")
    if not label_json.exists():
        raise FileNotFoundError(f"Label JSON not found: {label_json}")

    text_map = load_json(text_json)
    label_map = load_json(label_json)
    if not isinstance(text_map, dict):
        raise ValueError("text-json must be a dict in {image_path: text} format.")
    if not isinstance(label_map, dict):
        raise ValueError("label-json must be a dict in {image_path: label_id} format.")

    structured = {}
    original_format = {}
    label_counts = {str(label): 0 for label in sorted(LABEL_TO_DEFECT)}
    unknown_label_count = 0
    text_missing_count = 0
    welding_type_count = 0
    material_count = 0

    for key, label_id in label_map.items():
        prompt = text_map.get(key, "")
        if prompt == "":
            text_missing_count += 1
        try:
            label_id_int = int(label_id)
        except (TypeError, ValueError):
            label_id_int = None

        defect_type = LABEL_TO_DEFECT.get(label_id_int)
        if defect_type is None:
            unknown_label_count += 1
        else:
            label_counts[str(label_id_int)] += 1

        welding_type = find_welding_type(key, prompt)
        material = find_material(key, prompt)
        if welding_type is not None:
            welding_type_count += 1
        if material is not None:
            material_count += 1

        category_text = make_category_text(welding_type, material, defect_type)
        structured[key] = {
            "category_text": category_text,
            "welding_type": welding_type,
            "welding_material": material,
            "defect_label": label_id_int,
            "defect_type": defect_type,
            "original_text": prompt,
        }
        original_format[key] = category_text

    result = {
        "_meta": {
            "text_source": str(text_json),
            "label_source": str(label_json),
            "count": len(structured),
            "fields": ["welding_type", "welding_material", "defect_label", "defect_type"],
            "label_to_defect": LABEL_TO_DEFECT,
            "label_counts": label_counts,
            "unknown_label_count": unknown_label_count,
            "text_missing_count": text_missing_count,
            "coverage": {
                "welding_type": welding_type_count,
                "welding_material": material_count,
                "defect_type": len(structured) - unknown_label_count,
            },
        },
        "items": structured,
    }

    out_json = Path(args.out_json)
    out_original = Path(args.out_original_format_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_original.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(out_original, "w", encoding="utf-8") as f:
        json.dump(original_format, f, ensure_ascii=False, indent=2)

    print(f"Wrote structured JSON: {out_json}")
    print(f"Wrote original-format JSON: {out_original}")
    print(f"Records: {len(structured)}")
    print(f"Label counts: {label_counts}")
    print(f"Text missing: {text_missing_count}")
    print(f"Unknown labels: {unknown_label_count}")


if __name__ == "__main__":
    main()
