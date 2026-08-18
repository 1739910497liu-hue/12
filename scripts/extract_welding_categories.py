import argparse
import json
import re
from pathlib import Path


def clean_text(value):
    return re.sub(r"\s+", " ", str(value).replace("\\", "/")).strip()


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


def has_no_defect_phrase(text):
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in [
            r"\bno\s+(?:reported\s+|visible\s+)?defects?\b",
            r"\bminimal\s+defects?\b",
            r"\bgood\s+weld(?:\s+quality)?\b",
            r"\bweld\s+quality\s+was\s+assessed\s+as\s+good\b",
            r"\bresult(?:ing)?\s+weld\s+was\s+assessed\s+as\s+good\b",
            r"\bproper\s+fusion\b",
            r"\bsuccessfully\s+joined\b",
            r"\bhigh-quality\s+weld\b",
        ]
    )


def find_defect_type(path_text, prompt_text):
    path = clean_text(path_text).lower()
    prompt = clean_text(prompt_text).lower()
    text = f"{path} {prompt}"

    defect_patterns = [
        (r"lack\s+of\s+fusion", "lack of fusion"),
        (r"lack\s+of\s+penetration|incomplete\s+penetration", "lack of penetration"),
        (r"contamination|contaminat|effys\s+glue|added\s+slug", "contamination"),
        (r"\bporosity\b|pores?\b", "porosity"),
        (r"\bcracks?\b|cracking", "crack"),
        (r"\bundercut\b", "undercut"),
        (r"burn[-\s]?through", "burn through"),
        (r"slag\s+inclusion|slug\s+inclusion|\bslag\b|\bslug\b", "slag inclusion"),
    ]
    for pattern, label in defect_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    if has_no_defect_phrase(text):
        return "no defect"
    return None


def make_category_text(welding_type, material, defect_type):
    return (
        "Categories: "
        f"welding type {welding_type or 'unknown'}, "
        f"material {material or 'unknown'}, "
        f"defect type {defect_type or 'unknown'}."
    )


def main():
    parser = argparse.ArgumentParser(description="Extract welding type, material, and defect type from welding JSON.")
    parser.add_argument("--input", default=r"H:\数据集\JSON\text\train_text.json")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-original-format-json", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    out_json = Path(args.out_json) if args.out_json else input_path.with_name(input_path.stem + "_categories.json")
    out_original = (
        Path(args.out_original_format_json)
        if args.out_original_format_json
        else input_path.with_name(input_path.stem + "_categories_original_format.json")
    )

    items = load_items(input_path)
    structured = {}
    original_format = {}
    coverage = {
        "welding_type": 0,
        "welding_material": 0,
        "defect_type": 0,
    }
    defect_counts = {}
    welding_type_counts = {}
    material_counts = {}

    for key, prompt in items:
        welding_type = find_welding_type(key, prompt)
        material = find_material(key, prompt)
        defect_type = find_defect_type(key, prompt)
        category_text = make_category_text(welding_type, material, defect_type)

        if welding_type is not None:
            coverage["welding_type"] += 1
            welding_type_counts[welding_type] = welding_type_counts.get(welding_type, 0) + 1
        if material is not None:
            coverage["welding_material"] += 1
            material_counts[material] = material_counts.get(material, 0) + 1
        if defect_type is not None:
            coverage["defect_type"] += 1
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1

        structured[key] = {
            "category_text": category_text,
            "welding_type": welding_type,
            "welding_material": material,
            "defect_type": defect_type,
            "original_text": prompt,
        }
        original_format[key] = category_text

    result = {
        "_meta": {
            "source": str(input_path),
            "count": len(items),
            "fields": ["welding_type", "welding_material", "defect_type"],
            "coverage": coverage,
            "welding_type_counts": welding_type_counts,
            "material_counts": material_counts,
            "defect_counts": defect_counts,
        },
        "items": structured,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(out_original, "w", encoding="utf-8") as f:
        json.dump(original_format, f, ensure_ascii=False, indent=2)

    print(f"Input records: {len(items)}")
    print(f"Wrote structured JSON: {out_json}")
    print(f"Wrote original-format JSON: {out_original}")
    print("Coverage:", coverage)
    print("Defect counts:", defect_counts)


if __name__ == "__main__":
    main()
