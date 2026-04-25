from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup

PRICE_RE = re.compile(
    r"(?:(?:US\$|\$|USD)\s*(?P<a>[0-9][0-9,]*(?:\.[0-9]{1,4})?)|(?P<b>[0-9][0-9,]*(?:\.[0-9]{1,4})?)\s*(?:USD|US\s?dollars))",
    re.I,
)
PACK_RE = re.compile(
    r"(?P<size>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ug|µg|μg|microgram(?:s)?|mg|milligram(?:s)?|g|gram(?:s)?|kg|kilogram(?:s)?|ml|mL|milliliter(?:s)?|L|l|liter(?:s)?)\b",
    re.I,
)
SOLUTION_RE = re.compile(
    r"(?i)(?P<sol>(?:[0-9]+(?:\.[0-9]+)?\s*mM\s*(?:x|\*)\s*[0-9]+(?:\.[0-9]+)?\s*mL(?:\s*in\s*DMSO)?|[0-9]+(?:\.[0-9]+)?\s*mL\s*(?:x|\*)\s*[0-9]+(?:\.[0-9]+)?\s*mM(?:\s*\(?\s*in\s*DMSO\s*\)?)?))"
)
STOCK_RE = re.compile(
    r"(?i)(in\s*stock|out\s*of\s*stock|available|ships?\s*in\s*[^.;,|]{1,45}|[0-9]+[- ]?[0-9]*\s*(?:days|weeks)|backorder|preorder|ask|quote|required|from\s+[A-Za-z ]+\s+partner)"
)
NOISE_RE = re.compile(
    r"(?i)(free\s+shipping|orders?\s+over|minimum\s+order|cart\s*(?:total|subtotal)|basket\b|subtotal|checkout|coupon|promo|discount|shipping\s+threshold|handling\s+fee|tax\b|recently\s+added|sign\s+in\s+to\s+checkout|save\s+\d+%|price\s+match)"
)
REFERENCE_RE = re.compile(r"(?i)(reference\s+standard|analytical\s+standard|\(\s*standard\s*\)|standard\s+solution)")
CAS_RE_TEMPLATE = r"\bCAS(?:\s*(?:No\.?|Number|#))?\s*[:\-]?\s*{cas}\b|(?<!\d){cas}(?!\d)"
SEARCH_LIKE_RE = re.compile(r"(?i)(/search|catalogsearch|keyword=|search=|q=|query=|find\.cgi|search\.aspx|search\.html|/result|products\?search|Search\?)")
PRODUCT_HINT_RE = re.compile(r"(?i)(/compound/|/products?/|/product/|/item/|/shop/compound/|/natural/|\.html$|\.htm$)")
MW_CONTEXT_RE = re.compile(r"(?i)(molecular\s+weight|mw\b|m\.w\.|formula\s+weight|g\s*/\s*mol|g/mol|mol\b)")


@dataclass(frozen=True)
class SupplierParserProfile:
    supplier: str
    family: str
    row_markers: tuple[str, ...]
    pipeline: tuple[str, ...]
    allow_search_result_cards: bool = False
    notes: str = ""


PRICE_FIRST = "size_price_stock"
SPECIALTY = "specialty_table"
MARKETPLACE = "marketplace"
DISTRIBUTOR = "distributor_login_or_public"
DIRECTORY = "directory_leads"

BASE_PIPE = ("json", "attributes", "table", "option", "text")
PRICE_PIPE = ("json", "attributes", "table", "option", "text", "cards")
SPECIALTY_PIPE = ("json", "attributes", "table", "price_per_pack", "text", "cards")
DISTRIBUTOR_PIPE = ("json", "attributes", "table", "distributor_each", "text")
DIRECTORY_PIPE = ("lead_cards",)

SUPPLIER_PARSER_PROFILES: dict[str, SupplierParserProfile] = {
    "TargetMol": SupplierParserProfile("TargetMol", PRICE_FIRST, ("Pack Size", "Price", "USA Stock", "Global Stock", "Add to Cart"), PRICE_PIPE, True),
    "MedChemExpress": SupplierParserProfile("MedChemExpress", PRICE_FIRST, ("Size", "Price", "Stock", "Quantity", "Solid + Solvent"), PRICE_PIPE),
    "SelleckChem": SupplierParserProfile("SelleckChem", PRICE_FIRST, ("Size", "Price", "Stock", "Quantity"), PRICE_PIPE),
    "Cayman Chemical": SupplierParserProfile("Cayman Chemical", PRICE_FIRST, ("Item", "Qty", "Price", "Availability", "CAS Number"), ("json", "attributes", "table", "cayman", "text", "cards"), True),
    "MolPort": SupplierParserProfile("MolPort", MARKETPLACE, ("Available Packings", "Price", "Supplier", "Ships"), PRICE_PIPE),
    "Adooq": SupplierParserProfile("Adooq", PRICE_FIRST, ("Size", "Price", "Availability", "Quantity"), PRICE_PIPE),
    "ApexBio": SupplierParserProfile("ApexBio", PRICE_FIRST, ("Size", "Price", "Availability", "Quantity"), PRICE_PIPE),
    "GLP Bio": SupplierParserProfile("GLP Bio", PRICE_FIRST, ("Size", "Price", "Stock", "Availability"), PRICE_PIPE),
    "AbMole": SupplierParserProfile("AbMole", PRICE_FIRST, ("Size", "Price", "Availability", "Quantity"), PRICE_PIPE),
    "ChemFaces": SupplierParserProfile("ChemFaces", SPECIALTY, ("Price:", "Size /Price /Stock", "CAS No.", "Catalog No."), ("json", "attributes", "chemfaces", "price_per_pack", "text", "cards")),
    "BioCrick": SupplierParserProfile("BioCrick", SPECIALTY, ("Price", "Stock", "Size", "CAS"), SPECIALTY_PIPE),
    "CSNpharm": SupplierParserProfile("CSNpharm", PRICE_FIRST, ("Size", "Price", "Stock", "Quantity"), PRICE_PIPE),
    "InvivoChem": SupplierParserProfile("InvivoChem", PRICE_FIRST, ("Size", "Price", "Stock", "Quantity"), PRICE_PIPE),
    "AdooQ Bioscience": SupplierParserProfile("AdooQ Bioscience", PRICE_FIRST, ("Size", "Price", "Availability", "Quantity"), PRICE_PIPE),
    "Biorbyt": SupplierParserProfile("Biorbyt", SPECIALTY, ("Size", "Price", "Availability", "Stock"), SPECIALTY_PIPE),
    "US Biological": SupplierParserProfile("US Biological", SPECIALTY, ("Pricing", "List Price", "Sizes", "CAS"), SPECIALTY_PIPE, True),
    "Biosynth": SupplierParserProfile("Biosynth", SPECIALTY, ("Size", "Price", "Availability", "CAS"), SPECIALTY_PIPE),
    "1ClickChemistry": SupplierParserProfile("1ClickChemistry", SPECIALTY, ("Size", "Price", "Availability", "Pack"), SPECIALTY_PIPE),
    "TCI Chemicals": SupplierParserProfile("TCI Chemicals", SPECIALTY, ("Size", "Price", "Stock", "CAS RN"), SPECIALTY_PIPE),
    "Oakwood Chemical": SupplierParserProfile("Oakwood Chemical", SPECIALTY, ("Size", "Price", "Availability", "SKU"), SPECIALTY_PIPE),
    "Chem-Impex": SupplierParserProfile("Chem-Impex", SPECIALTY, ("Pack Size", "Price", "Availability", "Catalog"), SPECIALTY_PIPE),
    "Combi-Blocks": SupplierParserProfile("Combi-Blocks", SPECIALTY, ("Price", "Availability", "Catalog", "CAS"), SPECIALTY_PIPE),
    "BLD Pharm": SupplierParserProfile("BLD Pharm", SPECIALTY, ("Size", "Price", "Stock", "CAS"), SPECIALTY_PIPE),
    "Ambeed": SupplierParserProfile("Ambeed", MARKETPLACE, ("Size", "Price", "Stock", "Supplier", "Choose your location", "Ship to"), ("ambeed", "json", "attributes", "table", "option", "text", "cards")),
    "A2B Chem": SupplierParserProfile("A2B Chem", SPECIALTY, ("Size", "Price", "Availability", "CAS"), SPECIALTY_PIPE),
    "Enamine": SupplierParserProfile("Enamine", SPECIALTY, ("Pack", "Price", "Availability", "Catalogue"), SPECIALTY_PIPE),
    "Matrix Scientific": SupplierParserProfile("Matrix Scientific", SPECIALTY, ("Price", "Size", "Catalog", "CAS"), SPECIALTY_PIPE),
    "Santa Cruz Biotechnology": SupplierParserProfile("Santa Cruz Biotechnology", SPECIALTY, ("Size", "Price", "Availability", "CAS"), SPECIALTY_PIPE),
    "CymitQuimica": SupplierParserProfile("CymitQuimica", MARKETPLACE, ("Price", "Availability", "Quantity", "Supplier"), PRICE_PIPE),
    "Toronto Research Chemicals": SupplierParserProfile("Toronto Research Chemicals", SPECIALTY, ("Size", "Price", "Availability", "Catalogue"), SPECIALTY_PIPE),
    "Fisher Scientific": SupplierParserProfile("Fisher Scientific", DISTRIBUTOR, ("Sign In or Register to check your price", "Each of 1", "Catalog No.", "Supplier Partner"), DISTRIBUTOR_PIPE, True),
    "Thermo Fisher / Alfa Aesar": SupplierParserProfile("Thermo Fisher / Alfa Aesar", DISTRIBUTOR, ("Sign In", "Catalog number", "Price", "Each"), DISTRIBUTOR_PIPE, True),
    "Sigma-Aldrich": SupplierParserProfile("Sigma-Aldrich", DISTRIBUTOR, ("Pricing", "Availability", "Pack Size", "Select a Size", "Vendor SKU"), ("sigma", "json", "attributes", "table", "distributor_each", "text")),
    "VWR / Avantor": SupplierParserProfile("VWR / Avantor", DISTRIBUTOR, ("Price", "Pack", "Availability", "Login"), DISTRIBUTOR_PIPE, True),
    "ChemicalBook": SupplierParserProfile("ChemicalBook", DIRECTORY, ("Supplier", "CAS", "Price", "Quote"), DIRECTORY_PIPE, True),
    "ChemBlink": SupplierParserProfile("ChemBlink", DIRECTORY, ("Supplier", "CAS", "Price", "Quote"), DIRECTORY_PIPE, True),
    "ChemExper": SupplierParserProfile("ChemExper", DIRECTORY, ("Supplier", "Catalog", "CAS", "Price"), DIRECTORY_PIPE, True),
    "LookChem": SupplierParserProfile("LookChem", DIRECTORY, ("Supplier", "CAS", "Price", "Quote"), DIRECTORY_PIPE, True),
}


def _safe_float(value: Any) -> float | None:
    try:
        f = float(str(value).replace(",", "").replace("$", "").replace("USD", "").strip())
        if f > 0:
            return f
    except Exception:
        return None
    return None


def _normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    u = unit.strip().replace("μ", "u").replace("µ", "u").lower()
    mapping = {
        "microgram": "ug", "micrograms": "ug", "ug": "ug",
        "milligram": "mg", "milligrams": "mg", "mg": "mg",
        "gram": "g", "grams": "g", "g": "g",
        "kilogram": "kg", "kilograms": "kg", "kg": "kg",
        "milliliter": "mL", "milliliters": "mL", "ml": "mL",
        "liter": "L", "liters": "L", "l": "L",
    }
    return mapping.get(u, unit)


def _price_matches(text: str | None):
    return list(PRICE_RE.finditer(str(text or "")))


def _pack_matches(text: str | None):
    raw = str(text or "").replace("μ", "u").replace("µ", "u")
    out = []
    for m in PACK_RE.finditer(raw):
        start, end = m.span()
        ctx = raw[max(0, start - 55): min(len(raw), end + 55)]
        if MW_CONTEXT_RE.search(ctx):
            continue
        out.append(m)
    return out


def _parse_pack(text: str | None, price_span: tuple[int, int] | None = None) -> tuple[float | None, str | None]:
    if not text:
        return None, None
    raw = str(text).replace("μ", "u").replace("µ", "u")
    sol = SOLUTION_RE.search(raw)
    if sol:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mL", sol.group("sol"), re.I)
        if m:
            return _safe_float(m.group(1)), "mL"
    packs = _pack_matches(raw)
    if not packs:
        return None, None
    if price_span is not None:
        p_center = (price_span[0] + price_span[1]) / 2
        packs.sort(key=lambda m: abs(((m.start() + m.end()) / 2) - p_center))
    m = packs[0]
    return _safe_float(m.group("size")), _normalize_unit(m.group("unit"))


def _parse_price(text: str | None) -> float | None:
    m = PRICE_RE.search(str(text or ""))
    if not m:
        return None
    return _safe_float(m.group("a") or m.group("b"))


def _pack_ok(size: float | None, unit: str | None) -> bool:
    if size is None or unit is None:
        return False
    caps = {"ug": 1_000_000_000, "mg": 1_000_000, "g": 100_000, "kg": 10_000, "mL": 1_000_000, "L": 10_000}
    return 0 < float(size) <= caps.get(unit, caps.get(str(unit).lower(), 100_000))


def _stock(text: str | None) -> str:
    m = STOCK_RE.search(text or "")
    return re.sub(r"\s+", " ", m.group(1)).strip().title() if m else "Not visible"


def _is_noise(text: str | None) -> bool:
    return bool(NOISE_RE.search(text or ""))


def _product_form(text: str | None, unit: str | None) -> str:
    raw = text or ""
    if REFERENCE_RE.search(raw):
        return "standard/reference"
    if unit in {"mL", "L"}:
        return "solution"
    if unit in {"ug", "mg", "g", "kg"}:
        return "solid/mass"
    if SOLUTION_RE.search(raw):
        return "solution"
    return "unknown"


def _row(method: str, size: float | None, unit: str | None, price: float | None, raw: str, confidence: str = "HIGH", validation: str = "verified_product", lead_cas_match: bool = False) -> dict[str, Any] | None:
    raw = re.sub(r"\s+", " ", str(raw or "")).strip()
    if _is_noise(raw):
        return None
    if price is None or not _pack_ok(size, unit):
        return None
    return {
        "method": method,
        "pack_size": size,
        "pack_unit": unit,
        "price": price,
        "stock": _stock(raw),
        "raw": [raw[:1500]],
        "price_pairing_confidence": confidence,
        "product_form": _product_form(raw, unit),
        "price_validation_level": validation,
        "lead_cas_match": lead_cas_match,
    }


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    priority = {
        "supplier_specific_json_row": 8,
        "supplier_specific_attribute_row": 7,
        "supplier_specific_table_row": 6,
        "supplier_specific_size_price_row": 5,
        "supplier_specific_price_per_pack_row": 5,
        "supplier_specific_cayman_colon_row": 6,
        "supplier_specific_chemfaces_row": 6,
        "supplier_specific_sigma_pack_row": 6,
        "supplier_specific_distributor_each_row": 4,
        "supplier_specific_option_row": 4,
        "supplier_specific_search_card_lead": 2,
    }
    rows = sorted(rows, key=lambda r: priority.get(str(r.get("method", "")).split("_")[0], priority.get(str(r.get("method", "")).rsplit("_", 1)[0], 0)), reverse=True)
    for r in rows:
        key = (
            round(float(r.get("pack_size") or 0), 9),
            str(r.get("pack_unit") or "").lower(),
            round(float(r.get("price") or 0), 4),
            str(r.get("product_form") or ""),
            str(r.get("price_validation_level") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _table_rows(soup: BeautifulSoup) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if not cells:
                continue
            row_text = " | ".join(cells)
            if len(row_text) < 4 or _is_noise(row_text):
                continue
            price_m = PRICE_RE.search(row_text)
            price = _parse_price(row_text)
            size, unit = _parse_pack(row_text, price_m.span() if price_m else None)
            candidate = _row("supplier_specific_table_row", size, unit, price, row_text, "HIGH")
            if candidate:
                rows.append(candidate)
    return rows


def _option_rows(soup: BeautifulSoup) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag in soup.find_all(["option", "button", "input", "li", "div"], limit=5000):
        attrs = " ".join(f"{k}={v}" for k, v in tag.attrs.items() if isinstance(v, (str, int, float)))
        txt = tag.get_text(" ", strip=True)
        raw = f"{txt} {attrs}"
        if len(raw) < 4 or _is_noise(raw):
            continue
        price_m = PRICE_RE.search(raw)
        size, unit = _parse_pack(raw, price_m.span() if price_m else None)
        price = _parse_price(raw)
        candidate = _row("supplier_specific_option_row", size, unit, price, raw, "MEDIUM")
        if candidate:
            rows.append(candidate)
    return rows


def _attribute_rows(soup: BeautifulSoup) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag in soup.find_all(True, limit=7000):
        if not tag.attrs:
            continue
        raw_attrs = " ".join(f"{k}={v}" for k, v in tag.attrs.items())
        txt = tag.get_text(" ", strip=True)
        raw = f"{txt} {raw_attrs}"
        if len(raw) < 5 or _is_noise(raw):
            continue
        low_attrs = raw_attrs.lower()
        if not any(k in low_attrs for k in ["price", "amount", "sku", "pack", "size", "qty", "catalog"]):
            continue
        price = None
        for key in ["data-price", "data-unit-price", "data-final-price", "data-sale-price", "price", "data-amount"]:
            if tag.has_attr(key):
                price = _safe_float(tag.get(key))
                if price:
                    break
        if price is None:
            price = _parse_price(raw)
        price_m = PRICE_RE.search(raw)
        size, unit = _parse_pack(raw, price_m.span() if price_m else None)
        candidate = _row("supplier_specific_attribute_row", size, unit, price, raw, "HIGH")
        if candidate:
            rows.append(candidate)
    return rows


def _size_price_text_rows(text: str, profile: SupplierParserProfile) -> list[dict[str, Any]]:
    clean = re.sub(r"\s+", " ", (text or "").replace("μ", "u").replace("µ", "u"))
    if not clean:
        return []
    marker_locs = []
    for marker in profile.row_markers:
        for m in re.finditer(re.escape(marker), clean, re.I):
            marker_locs.append(m.start())
    windows = [clean[max(0, i - 800): i + 14000] for i in sorted(marker_locs)[:10]]
    if not windows and profile.family != DIRECTORY:
        windows = [clean[:42000]]
    rows: list[dict[str, Any]] = []
    size_token = r"(?:\d+(?:\.\d+)?\s?(?:ug|mg|g|kg|mL|ml|L)|\d+(?:\.\d+)?\s?mM\s*(?:x|\*)\s*\d+(?:\.\d+)?\s?mL(?:\s*in\s*DMSO)?|\d+(?:\.\d+)?\s?mL\s*(?:x|\*)\s*\d+(?:\.\d+)?\s?mM(?:\s*\(?\s*in\s*DMSO\s*\)?)?)"
    price_token = r"(?:(?:US\$|\$|USD)\s*[0-9][0-9,]*(?:\.\d{1,4})?|[0-9][0-9,]*(?:\.\d{1,4})?\s*USD)"
    stock_token = r"(?P<stock>in\s*stock|out\s*of\s*stock|available|ships?\s*in\s*[^.;,|]{1,35}|[0-9]+[- ]?[0-9]*\s*(?:days|weeks))?"
    patterns = [
        re.compile(rf"(?P<size>{size_token})\s*(?:[,/|;:·•\-]|\s)+(?P<price>{price_token})\s*[-,|/·• ]*{stock_token}", re.I),
        re.compile(rf"(?P<size>{size_token})(?P<price>\$\s*[0-9][0-9,]*(?:\.\d{{1,4}})?)\s*[-,|/·• ]*{stock_token}", re.I),
    ]
    for window in windows:
        for pat in patterns:
            for m in pat.finditer(window):
                raw = " ".join(x for x in [m.group("size"), m.group("price"), (m.groupdict().get("stock") or "")] if x)
                size, unit = _parse_pack(m.group("size"))
                price = _parse_price(m.group("price"))
                candidate = _row("supplier_specific_size_price_row", size, unit, price, raw, "HIGH")
                if candidate:
                    rows.append(candidate)
    return rows


def _price_per_pack_rows(text: str) -> list[dict[str, Any]]:
    clean = re.sub(r"\s+", " ", (text or "").replace("μ", "u").replace("µ", "u"))
    rows: list[dict[str, Any]] = []
    patterns = [
        re.compile(r"(?i)(?:price\s*[:：]?\s*)?(?P<price>(?:US\$|\$|USD)\s*[0-9][0-9,]*(?:\.\d{1,4})?)\s*/\s*(?P<size>[0-9]+(?:\.[0-9]+)?\s*(?:ug|mg|g|kg|mL|ml|L))"),
        re.compile(r"(?i)(?P<size>[0-9]+(?:\.[0-9]+)?\s*(?:ug|mg|g|kg|mL|ml|L))\s*/\s*(?P<price>(?:US\$|\$|USD)\s*[0-9][0-9,]*(?:\.\d{1,4})?)"),
    ]
    for pat in patterns:
        for m in pat.finditer(clean):
            raw = m.group(0)
            size, unit = _parse_pack(m.group("size"))
            price = _parse_price(m.group("price"))
            candidate = _row("supplier_specific_price_per_pack_row", size, unit, price, raw, "HIGH")
            if candidate:
                rows.append(candidate)
    return rows


def _chemfaces_rows(text: str) -> list[dict[str, Any]]:
    rows = _price_per_pack_rows(text)
    clean = re.sub(r"\s+", " ", (text or "").replace("μ", "u").replace("µ", "u"))
    pat = re.compile(r"(?i)(?P<size>\d+(?:\.\d+)?\s*mM\s*(?:\*|x)\s*\d+(?:\.\d+)?\s*mL\s*(?:in\s*DMSO)?)\s*/\s*(?P<price>\$\s*[0-9][0-9,]*(?:\.\d{1,4})?)\s*/\s*(?P<stock>In-stock|In stock|Available)")
    for m in pat.finditer(clean):
        size, unit = _parse_pack(m.group("size"))
        price = _parse_price(m.group("price"))
        candidate = _row("supplier_specific_chemfaces_row", size, unit, price, m.group(0), "HIGH")
        if candidate:
            rows.append(candidate)
    return rows


def _cayman_rows(text: str) -> list[dict[str, Any]]:
    clean = re.sub(r"\s+", " ", (text or "").replace("μ", "u").replace("µ", "u"))
    rows: list[dict[str, Any]] = []
    pat = re.compile(r"(?i)(?P<size>\d+(?:\.\d+)?\s*(?:ug|mg|g|kg|mL|ml|L))\s*[:|]\s*(?P<price>\$\s*[0-9][0-9,]*(?:\.\d{1,4})?)\s*[:|]?\s*(?P<stock>In\s*stock|Available|Ships?\s*in\s*[^.;,|]{1,35})?")
    for m in pat.finditer(clean):
        raw = m.group(0)
        size, unit = _parse_pack(m.group("size"))
        price = _parse_price(m.group("price"))
        candidate = _row("supplier_specific_cayman_colon_row", size, unit, price, raw, "HIGH")
        if candidate:
            rows.append(candidate)
    return rows


def _jsonish_texts(soup: BeautifulSoup) -> Iterable[str]:
    for script in soup.find_all("script"):
        txt = script.get_text(" ", strip=True)
        if txt and ("price" in txt.lower() or "$" in txt or "sku" in txt.lower() or "pack" in txt.lower()):
            yield txt[:250000]


def _embedded_json_rows(soup: BeautifulSoup) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for txt in _jsonish_texts(soup):
        clean = re.sub(r"\s+", " ", txt.replace("\\/", "/").replace("μ", "u").replace("µ", "u"))
        for price_m in _price_matches(clean):
            start, end = price_m.span()
            window = clean[max(0, start - 500): min(len(clean), end + 500)]
            if _is_noise(window) or not re.search(r"(?i)(sku|pack|size|qty|catalog|availability|stock|price)", window):
                continue
            size, unit = _parse_pack(window, (min(500, start), min(500, end)))
            price = _safe_float(price_m.group("a") or price_m.group("b"))
            candidate = _row("supplier_specific_json_row", size, unit, price, window, "HIGH")
            if candidate:
                rows.append(candidate)
    return rows


def _sigma_pack_rows(text: str) -> list[dict[str, Any]]:
    clean = re.sub(r"\s+", " ", (text or "").replace("μ", "u").replace("µ", "u"))
    rows: list[dict[str, Any]] = []
    pat = re.compile(
        r"(?i)(?P<size>\d+(?:\.\d+)?\s*(?:ug|mg|g|kg|mL|ml|L))\s+(?P<sku>[A-Z0-9][A-Z0-9._-]{3,60})\s+(?P<availability>Ships?\s+in\s+[^$]{1,90}|Available|In\s*stock)?\s*(?P<price>\$\s*[0-9][0-9,]*(?:\.\d{1,4})?)"
    )
    for m in pat.finditer(clean):
        raw = m.group(0)
        size, unit = _parse_pack(m.group("size"))
        price = _parse_price(m.group("price"))
        candidate = _row("supplier_specific_sigma_pack_row", size, unit, price, raw, "HIGH")
        if candidate:
            rows.append(candidate)
    return rows


def _distributor_each_rows(title: str, text: str) -> list[dict[str, Any]]:
    hay = re.sub(r"\s+", " ", f"{title} {text[:50000]}").replace("μ", "u").replace("µ", "u")
    rows: list[dict[str, Any]] = []
    for m in re.finditer(r"(?P<price>\$\s*[0-9][0-9,]*(?:\.\d{1,4})?)\s*/\s*(?:Each|EA|Pack|Pkg)\b", hay, re.I):
        raw = hay[max(0, m.start() - 600): min(len(hay), m.end() + 350)]
        price = _parse_price(m.group("price"))
        size, unit = _parse_pack(raw, (min(600, m.start()), min(600, m.end())))
        candidate = _row("supplier_specific_distributor_each_row", size, unit, price, raw, "MEDIUM")
        if candidate:
            rows.append(candidate)
    return rows


def _card_contexts_with_cas(soup: BeautifulSoup, cas_number: str) -> list[str]:
    contexts: list[str] = []
    cas_re = re.compile(CAS_RE_TEMPLATE.format(cas=re.escape(cas_number)), re.I)
    for tag in soup.find_all(string=cas_re):
        node = getattr(tag, "parent", None)
        for parent in [node, node.find_parent("tr") if node else None, node.find_parent("li") if node else None, node.find_parent("div") if node else None]:
            if parent is None:
                continue
            txt = parent.get_text(" ", strip=True)
            if txt and txt not in contexts:
                contexts.append(txt[:2500])
    return contexts[:25]


def _search_card_lead_rows(soup: BeautifulSoup, text: str, cas_number: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contexts = _card_contexts_with_cas(soup, cas_number)
    if not contexts and cas_number in (text or ""):
        for m in re.finditer(re.escape(cas_number), text or "", re.I):
            contexts.append((text or "")[max(0, m.start() - 900): min(len(text or ""), m.end() + 1200)])
            if len(contexts) >= 15:
                break
    for ctx in contexts:
        clean = re.sub(r"\s+", " ", ctx.replace("μ", "u").replace("µ", "u"))
        if _is_noise(clean):
            continue
        for price_m in _price_matches(clean):
            price = _safe_float(price_m.group("a") or price_m.group("b"))
            size, unit = _parse_pack(clean, price_m.span())
            candidate = _row("supplier_specific_search_card_lead", size, unit, price, clean, "LOW", "tentative_search_or_directory", True)
            if candidate:
                rows.append(candidate)
    return rows


def _url_is_search_like(url: str | None, title: str | None = None) -> bool:
    hay = f"{url or ''} {title or ''}"
    return bool(SEARCH_LIKE_RE.search(hay) or re.search(r"(?i)search results|results page", hay))


def _profile_for(supplier: str, url: str | None = None) -> SupplierParserProfile | None:
    profile = SUPPLIER_PARSER_PROFILES.get(supplier)
    if profile is not None:
        return profile
    host = urlparse(url or "").netloc.lower()
    for p in SUPPLIER_PARSER_PROFILES.values():
        stem = p.supplier.lower().split()[0].replace("/", "")
        if stem and stem in host:
            return p
    return None


def parse_supplier_specific_rows(
    supplier: str,
    soup: BeautifulSoup,
    text: str,
    url: str,
    cas_number: str,
    title: str | None = None,
) -> list[dict[str, Any]]:
    profile = _profile_for(supplier, url)
    if profile is None:
        return []
    if profile.family == DIRECTORY:
        return []
    cas_pattern = re.compile(CAS_RE_TEMPLATE.format(cas=re.escape(cas_number)), re.I)
    cas_blob = " ".join([title or "", url or "", (text or "")[:70000]])
    if not cas_pattern.search(cas_blob):
        return []

    rows: list[dict[str, Any]] = []
    pipeline = profile.pipeline
    if "json" in pipeline:
        rows.extend(_embedded_json_rows(soup))
    if "attributes" in pipeline:
        rows.extend(_attribute_rows(soup))
    if "table" in pipeline:
        rows.extend(_table_rows(soup))
    if "option" in pipeline:
        rows.extend(_option_rows(soup))
    if "price_per_pack" in pipeline:
        rows.extend(_price_per_pack_rows(text))
    if "chemfaces" in pipeline:
        rows.extend(_chemfaces_rows(text))
    if "cayman" in pipeline:
        rows.extend(_cayman_rows(text))
    if "sigma" in pipeline:
        rows.extend(_sigma_pack_rows(text))
    if "distributor_each" in pipeline:
        rows.extend(_distributor_each_rows(title or "", text))
    if "ambeed" in pipeline:
        # Ambeed can show a country/location chooser before prices. The same page often still ships
        # JSON/DOM price fragments after a USA/default-location cookie; parse all robust fragments first.
        rows.extend(_sigma_pack_rows(text))
    if "text" in pipeline:
        rows.extend(_size_price_text_rows(text, profile))
    if "cards" in pipeline and not rows:
        rows.extend(_search_card_lead_rows(soup, text, cas_number))
        for row in rows:
            row["price_pairing_confidence"] = "LOW"
            row["price_validation_level"] = "tentative_card_on_product_page"

    for row in rows:
        row["method"] = f"{row.get('method')}_{profile.supplier.replace(' ', '_').replace('/', '').lower()}"
        row["supplier_parser_family"] = profile.family
        row["supplier_parser_name"] = profile.supplier
    return _dedupe(rows)


def parse_supplier_lead_rows(
    supplier: str,
    soup: BeautifulSoup,
    text: str,
    url: str,
    cas_number: str,
    title: str | None = None,
) -> list[dict[str, Any]]:
    profile = _profile_for(supplier, url)
    if profile is None:
        return []
    if not (_url_is_search_like(url, title) or profile.family == DIRECTORY or profile.allow_search_result_cards):
        return []
    rows = _search_card_lead_rows(soup, text, cas_number)
    for row in rows:
        row["method"] = f"{row.get('method')}_{profile.supplier.replace(' ', '_').replace('/', '').lower()}"
        row["supplier_parser_family"] = profile.family
        row["supplier_parser_name"] = profile.supplier
    return _dedupe(rows)
