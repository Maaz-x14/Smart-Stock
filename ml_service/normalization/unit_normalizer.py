# ml_service/normalization/unit_normalizer.py

import re
from dataclasses import dataclass

UNIT_MAP = {
    # Weight
    "LB":    "lb",  "LBS":   "lb"   ,  "POUND":     "lb" ,  "POUNDS":     "lb",
    "OZ":    "oz",  "OUNCE": "oz"   ,  "OUNCES":    "oz" ,  "OUNC": "oz", "OUNCES": "oz",
    "KG":    "kg",  "KGS":   "kg"   ,  "KILO":      "kg" ,  "KILOS":      "kg", "KILOGRAM": "kg", "KILOGRAMS": "kg",
    "G":     "g" ,  "GR":    "g"    ,  "GRM":       "g"  ,  "GRAM":       "g",   "GRAMS": "g", "GRAMME": "g",
    "MG":    "mg",  "MGS":   "mg"   ,  "MILLIGRAM": "mg" ,  "MILLIGRAMS": "mg",
    "MCG": "mcg" ,  "UG":    "mcg"  ,  "MICROGRAM": "mcg",
    "ST": "stone",  "STONE": "stone",
    
    # Volume
    "GAL":   "gal", "GL":   "gal", "GALLON": "gal", "GALLONS": "gal",
    "QT":    "qt",  "QUART": "qt",
    "PT":    "pt",  "PINT": "pt",
    "L":     "l",   "LTR":  "l",   "LITRE": "l",   "LITER":  "l", "LTRS": "l", "LTRES": "l",
    "ML":    "ml",  "MLS":  "ml",
    "FL OZ": "fl_oz",
    "CL":    "cl", "CENTILITER": "cl", "CENTILITRE": "cl",
    "DL":    "dl", "DECILITER": "dl", "DECILITRE": "dl",
    "TBSP":  "tbsp", "TBLSPN": "tbsp", "TABLESPOON": "tbsp", "TABLESPOONS": "tbsp",
    "TSP":   "tsp", "TEASP": "tsp", "TEASPOON": "tsp", "TEASPOONS": "tsp",
    "CUP":   "cup", "CUPS": "cup", "CP": "cup",
    
    # Count/Pack
    "CT":    "count", "CNT": "count", "COUNT": "count",
    "PK":    "pack",  "PCK": "pack",  "PACK":  "pack",
    "EA":    "each",  "PC":  "each",  "PCS":   "each",
    "DOZ":   "dozen", "DZ":  "dozen", "DOZEN": "dozen",
    "BX":    "box",   "BOX": "box",
    "BAG":   "bag",   "BG":  "bag",
    "BTL":   "bottle","BT":  "bottle","BOTTLE":"bottle",
    "CAN":   "can",   "CN":  "can",
    "JAR":   "jar",
    "TUB":   "tub",
    "ROLL": "roll", "RL": "roll",
    "SLICE": "slice", "SLC": "slice",
    "STICK": "stick", "STK": "stick",
    "PAIR": "pair", "PR": "pair",
    "SET": "set",
    
    # Containers
    "CARTON": "carton", "CTN": "carton",
    "PACKET": "packet", "PKT": "packet",
    "SACHET": "sachet", "SACH": "sachet",
    "BTLS": "bottle", "BOTTLES": "bottle", "BTTL": "bottle",
    "CANS": "can", "JUG": "jug", "TIN": "can",
}


@dataclass
class ParsedQuantity:
    quantity: float
    unit: str | None


def parse_quantity_unit(raw_qty: str | None, raw_unit: str | None) -> ParsedQuantity:
    """
    Parse and normalize quantity and unit from NER output.

    Handles three common patterns:
    1. Separate tokens:   raw_qty="1", raw_unit="LB"   → quantity=1.0, unit="lb"
    2. Fused token (qty): raw_qty="1LB", raw_unit=None → quantity=1.0, unit="lb"
    3. Complex quantity:  raw_qty="2", raw_unit="X 12 OZ" (multipack) → quantity=24.0, unit="oz"

    Returns ParsedQuantity(quantity, unit).
    """
    qty_str  = (raw_qty  or "").upper().strip()
    unit_str = (raw_unit or "").upper().strip()

    # Case 2: fused token — e.g. "1LB", "500G", "2GAL"
    fused_match = re.match(r'^(\d+\.?\d*)\s*([A-Z]+)$', qty_str)
    if fused_match and not unit_str:
        qty_val   = float(fused_match.group(1))
        unit_raw  = fused_match.group(2)
        unit_norm = UNIT_MAP.get(unit_raw)
        return ParsedQuantity(quantity=qty_val, unit=unit_norm)

    # Case 3: multipack — e.g. qty="2", unit="X 12 OZ" → 2 × 12 = 24 oz
    multipack_match = re.match(r'^X?\s*(\d+\.?\d*)\s*([A-Z]+)$', unit_str)
    if multipack_match:
        try:
            base_qty  = float(qty_str) if qty_str else 1.0
            pack_qty  = float(multipack_match.group(1))
            unit_raw  = multipack_match.group(2)
            unit_norm = UNIT_MAP.get(unit_raw)
            return ParsedQuantity(quantity=base_qty * pack_qty, unit=unit_norm)
        except (ValueError, TypeError):
            pass

    # Case 1: standard separate tokens
    try:
        qty_val = float(qty_str) if qty_str else 1.0
    except ValueError:
        qty_val = 1.0

    unit_norm = UNIT_MAP.get(unit_str) if unit_str else None

    return ParsedQuantity(quantity=qty_val, unit=unit_norm)