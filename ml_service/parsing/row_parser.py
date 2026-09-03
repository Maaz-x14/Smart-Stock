import re
from rapidfuzz import fuzz
from ml_service.parsing.prefilter import should_drop_row

COLUMN_SYNONYMS = {
    "QUANTITY": ["qty", "quantity", "qty/wt"],
    "PRICE": ["price", "rate", "m.r.p", "mrp"],
    "TOTAL": ["total", "amount"],
    "DISCOUNT": ["discount", "disc", "disc. amount"],
    "ITEM": ["description", "item", "product", "item description"],
}


def classify_header_token(token, threshold=75):
    token_low = token.lower()
    matches = []
    for col_type, synonyms in COLUMN_SYNONYMS.items():
        best = max(fuzz.ratio(token_low, syn) for syn in synonyms)
        if best >= threshold:
            matches.append((col_type, best))

    if matches:
        return matches

    for col_type, synonyms in COLUMN_SYNONYMS.items():
        best = max(fuzz.partial_ratio(token_low, syn) for syn in synonyms)
        if best >= threshold:
            matches.append((col_type, best))
    return matches


def find_header_row(rows, max_scan=None):
    limit = len(rows)
    for idx, row in enumerate(rows):
        line = " ".join(item["text"] for item in row).upper()
        if "SALES ITEM" in line or line.strip() == "SALES":
            limit = idx
            break
    limit = min(limit, max_scan) if max_scan else limit

    best_row_idx, best_col_map, best_hits = None, None, 0
    for row_idx, row in enumerate(rows[:limit]):
        col_hits = 0
        col_map = []
        for item in row:
            matches = classify_header_token(item["text"])
            if matches:
                col_hits += 1
                col_map.append((item["x"], [m[0] for m in matches]))
        col_types_found = {t for _, types in col_map for t in types}
        required = {"QUANTITY", "PRICE", "TOTAL"}
        if col_hits >= 2 and len(required & col_types_found) >= 2 and col_hits > best_hits:
            best_row_idx, best_col_map, best_hits = row_idx, col_map, col_hits

    return best_row_idx, best_col_map

def clean_numeric_string(text):
    t = text.strip()

    # strip Rs/rs price prefix only
    t = re.sub(r"^[Rr]s\.?\s*", "", t)

    # reject if any letters remain — real numeric tokens don't have letters
    if re.search(r"[A-Za-z]", t):
        return None

    # now safe to strip stray OCR noise like ^, stray symbols
    t = re.sub(r"[^\d.,]", "", t)
    if not t:
        return None

    if t.count(".") > 1:
        parts = t.split(".")
        t = "".join(parts[:-1]) + "." + parts[-1]
    t = t.replace(",", "")

    try:
        return float(t)
    except ValueError:
        return None


def is_numeric(text):
    return clean_numeric_string(text) is not None


def to_number(text):
    return clean_numeric_string(text)

def parse_row_with_header(row, col_map):
    result = {"item_name": None, "quantity": None, "price": None,
              "discount": None, "total": None, "low_confidence_parse": False}
    for item in row:
        nearest = min(col_map, key=lambda c: abs(c[0] - item["x"]))
        col_types = nearest[1]

        if "ITEM" in col_types and len(col_types) == 1:
            result["item_name"] = item["text"]
        elif is_numeric(item["text"]):
            val = to_number(item["text"])
            if "QUANTITY" in col_types and "ITEM" in col_types:
                m = re.match(r"^(\d+\.?\d*)\s+(.+)", item["text"])
                if m:
                    result["quantity"] = float(m.group(1))
                    result["item_name"] = m.group(2)
                continue

            for col_type in col_types:
                if col_type == "QUANTITY" and result["quantity"] is None:
                    result["quantity"] = val
                    break
                elif col_type == "PRICE" and result["price"] is None:
                    result["price"] = val
                    break
                elif col_type == "DISCOUNT" and result["discount"] is None:
                    result["discount"] = val
                    break
                elif col_type == "TOTAL" and result["total"] is None:
                    result["total"] = val
                    break
            if len(col_types) > 1:
                result["low_confidence_parse"] = True
        else:
            if result["item_name"] is None:
                result["item_name"] = item["text"]
    return result


def parse_row_without_header(row):
    result = {"item_name": None, "quantity": None, "price": None,
              "discount": None, "total": None, "low_confidence_parse": True}

    non_numeric = [i for i in row if not is_numeric(i["text"])]
    numeric = [i for i in row if is_numeric(i["text"])]

    if non_numeric:
        result["item_name"] = non_numeric[0]["text"]

    nums = [(to_number(i["text"]), i["x"]) for i in numeric]
    nums.sort(key=lambda n: n[1])

    if len(nums) >= 1:
        result["total"] = nums[-1][0]
    if len(nums) >= 2:
        remaining = nums[:-1]
        int_like = [n for n in remaining if n[0] == int(n[0]) and n[0] < 100]
        if int_like:
            qty_candidate = min(int_like, key=lambda n: n[0])
            result["quantity"] = qty_candidate[0]
            remaining = [n for n in remaining if n != qty_candidate]
        if remaining:
            price_candidate = max(remaining, key=lambda n: n[0])
            result["price"] = price_candidate[0]
            remaining.remove(price_candidate)
        if remaining:
            result["discount"] = remaining[0][0]

    return result


def merge_split_rows(parsed_items):
    merged = []
    i = 0
    while i < len(parsed_items):
        item = parsed_items[i]
        has_name = item["item_name"] is not None
        has_numbers = any(item[k] is not None for k in ("quantity", "price", "discount", "total"))

        if has_name and has_numbers:
            merged.append(item)
            i += 1
        elif has_name and not has_numbers:
            if i + 1 < len(parsed_items):
                nxt = parsed_items[i + 1]
                nxt_has_name = nxt["item_name"] is not None
                nxt_has_numbers = any(nxt[k] is not None for k in ("quantity", "price", "discount", "total"))
                if not nxt_has_name and nxt_has_numbers:
                    combined = {
                        "item_name": item["item_name"],
                        "quantity": nxt["quantity"],
                        "price": nxt["price"],
                        "discount": nxt["discount"],
                        "total": nxt["total"],
                        "low_confidence_parse": item["low_confidence_parse"] or nxt["low_confidence_parse"],
                    }
                    merged.append(combined)
                    i += 2
                    continue
            merged.append(item)
            i += 1
        else:
            merged.append(item)
            i += 1
    return merged


def parse_receipt_rows(rows):
    header_idx, col_map = find_header_row(rows)
    parsed = []
    for idx, row in enumerate(rows):
        if idx == header_idx:
            continue
        if should_drop_row(row):
            continue
        if col_map is not None:
            parsed.append(parse_row_with_header(row, col_map))
        else:
            parsed.append(parse_row_without_header(row))

    merged = merge_split_rows(parsed)

    # New issue (filed this session): drop items where item_name is
    # None/empty after parsing. should_drop_row() (Stage 1.6 prefilter)
    # only screens raw OCR rows BEFORE parsing - it can't catch cases
    # where parsing itself produces an empty item_name (e.g. a row
    # that was all-numeric, or noise that consumed the only
    # non-numeric token). Those were reaching Stage 2 as "" and
    # burning an is_food classification call on nothing.
    return [item for item in merged if item["item_name"] and item["item_name"].strip()]
