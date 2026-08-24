import re

DROP_KEYWORDS = {
    "GST", "INVOICE", "TRANSACTION", "POS", "CUSTOMER", "CNIC", "PAYMENTS",
    "TOTAL", "DISCOUNT", "ROUNDING", "TAX BREAKUP", "MRP", "NON MRP",
    "CHANGE DUE", "CASH", "THANK YOU", "VISIT AGAIN", "COME AGAIN",
    "CASHIER", "OPERATOR", "SAVE RECEIPT", "SUBTOTAL", "VAT",
}

PRICE_ONLY_RE   = re.compile(r"^Rs?\.?\s*[\d,]+\.?\d*$", re.IGNORECASE)
PHONE_RE        = re.compile(r"^\+?\d[\d\-\s]{7,}$")
DATE_RE         = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
TIME_RE         = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?$", re.IGNORECASE)
SEPARATOR_RE    = re.compile(r"^[-=*_]{3,}$")
PERCENT_ONLY_RE = re.compile(r"^\d{1,2}(\.\d+)?\s*%$")
BARCODE_RE      = re.compile(r"^\d{8,}$")

def should_drop_line(line: str) -> bool:
    stripped = line.strip()

    if len(stripped) < 3:
        return True
    if any(kw in stripped.upper() for kw in DROP_KEYWORDS):
        return True
    if "@" in stripped or "www." in stripped.lower() or "http" in stripped.lower() or ".com" in stripped.lower():
        return True
    for pattern in (PRICE_ONLY_RE, PHONE_RE, DATE_RE, TIME_RE, SEPARATOR_RE, PERCENT_ONLY_RE, BARCODE_RE):
        if pattern.match(stripped):
            return True
    return False