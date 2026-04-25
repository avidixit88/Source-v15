from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, unquote
import re

PRICE_PUBLIC = "Public price extracted"
PRICE_SNIPPET = "Search-snippet price only"
PRICE_CANDIDATE = "Candidate price; CAS unconfirmed"
PRICE_LOCATION = "Location/country selection required"
PRICE_LOGIN = "Login/account price required"
PRICE_QUOTE = "Quote required"
PRICE_HIDDEN = "No public price detected"
PRICE_FAILED = "Extraction failed"
PRICE_RE = re.compile(r"(?:(?:US\$|\$|USD)\s?([0-9]{1,6}(?:,[0-9]{3})*(?:\.[0-9]{1,4})?|[0-9]+(?:\.[0-9]{1,4})?)|\b([0-9]{1,6}(?:,[0-9]{3})*(?:\.[0-9]{1,4})?)\s?(?:USD|US\s?dollars)\b)", re.I)
QUOTE_RE = re.compile(r"(?i)(request\s+a?\s*quote|ask\s+for\s+quotation|quote\s+only|bulk\s+inquiry|pricing\s+on\s+request|price\s+on\s+request|please\s+inquire|inquiry\s+price|inquire)")
LOGIN_RE = re.compile(r"(?i)(sign\s*in\s*(?:or\s*register)?\s*to\s*(?:check|view|see)\s*(?:your\s*)?price|login\s*to\s*view\s*price|log\s*in\s*to\s*view\s*price|account\s*specific\s*price|your\s*price)")
CATALOG_PATTERNS = [
    re.compile(r"\b(?:catalog|cat\.?|sku|item|part|product(?:\s*id)?)\s*(?:no\.?|number|#)?\s*[:#-]\s*([A-Z0-9][A-Z0-9._/-]{3,35})\b", re.I),
    re.compile(r"\b(?:catalog|cat\.?|sku|item|part|product(?:\s*id)?)\s*(?:no\.?|number|#)?\s+([A-Z0-9][A-Z0-9._/-]{3,35})\b", re.I),
]
BAD_CATALOG_TOKENS = {
    "library", "introduction", "analysis", "alysis", "search", "result", "results",
    "product", "products", "compound", "compounds", "catalog", "category", "price", "stock",
    "availability", "phillyrin", "forsythin", "standard", "details", "chemical", "cas",
    "home", "html", "login", "register", "contact", "cart", "checkout", "shipping",
}
LANG_PREFIXES = {"us", "en", "jp", "kr", "de", "fr", "sp", "es", "cn", "uk", "eu", "ca", "au"}
SEARCH_QUERY_KEYS = {"keyword", "search", "q", "text", "term", "query", "searchdto.searchparam", "utm_source", "utm_medium", "utm_campaign", "srsltid"}

@dataclass(frozen=True)
class SupplierAdapter:
    name: str
    domains: tuple[str, ...]
    search_url_templates: tuple[str, ...]
    notes: str
    public_price_likelihood: str = "mixed"
    search_priority: int = 50
    source_tier: str = "standard"
    expected_behavior: str = "mixed_public_or_quote"

    def matches(self, url: str) -> bool:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return any(host == d or host.endswith("." + d) or d in host for d in self.domains)

# v15 registry: this is the source of truth. We walk it supplier-by-supplier instead of hoping broad search order is correct.
ADAPTERS: tuple[SupplierAdapter, ...] = (
    SupplierAdapter("TargetMol", ("targetmol.com",), ("https://www.targetmol.com/search?keyword={cas}",), "Often exposes Pack Size / Price / USA Stock / Global Stock tables.", "high", 130, "price_first", "public_price_common"),
    SupplierAdapter("MedChemExpress", ("medchemexpress.com",), ("https://www.medchemexpress.com/cas/{cas}.html", "https://www.medchemexpress.com/search.html?q={cas}"), "Often exposes Size / Price / Stock tables.", "high", 128, "price_first", "public_price_common"),
    SupplierAdapter("SelleckChem", ("selleckchem.com",), ("https://www.selleckchem.com/search.html?searchDTO.searchParam={cas}",), "Bioactive catalog with frequent public size/price/stock rows.", "high", 126, "price_first", "public_price_common"),
    SupplierAdapter("Cayman Chemical", ("caymanchem.com",), ("https://www.caymanchem.com/search?q={cas}", "https://www.caymanchem.com/search?text={cas}", "https://www.caymanchem.com/product?keyword={cas}"), "Often exposes pack sizes, pricing, availability on product/item pages.", "high", 124, "price_first", "public_price_common"),
    SupplierAdapter("MolPort", ("molport.com",), ("https://www.molport.com/shop/find-chemicals-by-cas-number/{cas}",), "Marketplace with pack/price potential; may require JS/API for full data.", "high", 122, "marketplace", "marketplace_public_mixed"),
    SupplierAdapter("Adooq", ("adooq.com",), ("https://www.adooq.com/search?q={cas}", "https://www.adooq.com/catalogsearch/result/?q={cas}"), "Magento-style catalog; many pages show size/price/stock.", "high", 120, "price_first", "public_price_common"),
    SupplierAdapter("ApexBio", ("apexbt.com",), ("https://www.apexbt.com/search?q={cas}", "https://www.apexbt.com/catalogsearch/result/?q={cas}"), "Public size/price/stock rows on many compound pages.", "high", 118, "price_first", "public_price_common"),
    SupplierAdapter("GLP Bio", ("glpbio.com",), ("https://www.glpbio.com/search?q={cas}", "https://www.glpbio.com/catalogsearch/result/?q={cas}"), "Public size/price/stock common; multilingual mirrors must be deduped.", "high", 116, "price_first", "public_price_common"),
    SupplierAdapter("AbMole", ("abmole.com",), ("https://www.abmole.com/search?q={cas}", "https://www.abmole.com/search?search={cas}"), "Useful public catalog, but related-product pages can cause false CAS matches unless identity gated.", "high", 114, "price_first", "public_price_common"),
    SupplierAdapter("ChemFaces", ("chemfaces.com",), ("https://www.chemfaces.com/search/?q={cas}", "https://www.chemfaces.com/search?q={cas}", "https://www.chemfaces.com/search.php?key={cas}", "https://www.chemfaces.com/search.php?keywords={cas}"), "Natural-products/reference supplier with many public prices.", "high", 112, "price_first", "public_price_common"),
    SupplierAdapter("BioCrick", ("biocrick.com",), ("https://www.biocrick.com/search?keyword={cas}",), "Natural products catalog; public price rows appear on many pages.", "medium", 110, "price_first", "public_price_mixed"),
    SupplierAdapter("CSNpharm", ("csnpharm.com",), ("https://csnpharm.com/search?q={cas}",), "Often has size/price/stock tables; some values encoded or login-dependent.", "medium", 108, "price_first", "public_price_mixed"),
    SupplierAdapter("InvivoChem", ("invivochem.com",), ("https://www.invivochem.com/search?q={cas}", "https://www.invivochem.com/search?search={cas}"), "Often lists sizes; prices may be hidden for many products.", "medium", 106, "price_first", "public_price_mixed"),
    SupplierAdapter("AdooQ Bioscience", ("adooqbioscience.com",), ("https://www.adooqbioscience.com/catalogsearch/result/?q={cas}",), "Alternate AdooQ domain seen in some search results.", "medium", 104, "price_first", "public_price_mixed"),
    SupplierAdapter("Biorbyt", ("biorbyt.com",), ("https://www.biorbyt.com/search?q={cas}",), "Research reagent supplier; public pricing varies by product/region.", "medium", 102, "price_first", "public_price_mixed"),
    SupplierAdapter("US Biological", ("usbio.net",), ("https://www.usbio.net/search?q={cas}", "https://www.usbio.net/biochemicals?search={cas}"), "Public list pricing is often visible for biochemical/reference products.", "medium", 101, "price_first", "public_price_mixed"),
    SupplierAdapter("Biosynth", ("biosynth.com",), ("https://www.biosynth.com/search?query={cas}", "https://www.biosynth.com/search?q={cas}"), "Specialty/reference supplier; public pricing varies by country and product.", "medium", 100, "price_first", "public_price_mixed"),
    SupplierAdapter("1ClickChemistry", ("1clickchemistry.com",), ("https://www.1clickchemistry.com/search?search={cas}", "https://www.1clickchemistry.com/search?q={cas}"), "Specialty chemical storefront with public price/availability language on some products.", "medium", 99, "price_first", "public_price_mixed"),

    SupplierAdapter("TCI Chemicals", ("tcichemicals.com",), ("https://www.tcichemicals.com/US/en/search?text={cas}",), "Reagent catalog; public pricing varies by region/session.", "medium", 96, "standard", "public_or_session_mixed"),
    SupplierAdapter("Oakwood Chemical", ("oakwoodchemical.com",), ("https://oakwoodchemical.com/Search?term={cas}",), "Specialty chemical catalog; mixed public price/quote behavior.", "medium", 94, "standard", "public_or_quote_mixed"),
    SupplierAdapter("Chem-Impex", ("chemimpex.com",), ("https://www.chemimpex.com/search?search={cas}",), "Specialty catalog; public/quote mixed. Needs strict product identity gating.", "medium", 92, "standard", "public_or_quote_mixed"),
    SupplierAdapter("Combi-Blocks", ("combi-blocks.com",), ("https://www.combi-blocks.com/cgi-bin/find.cgi?search={cas}",), "Building-block catalog; pricing may be public, quote, or login dependent.", "medium", 90, "standard", "public_or_quote_mixed"),
    SupplierAdapter("BLD Pharm", ("bldpharm.com",), ("https://www.bldpharm.com/search?search={cas}",), "Building-block catalog for advanced intermediates.", "medium", 88, "standard", "public_or_quote_mixed"),
    SupplierAdapter("Ambeed", ("ambeed.com",), ("https://www.ambeed.com/search.html?search={cas}", "https://www.ambeed.com/search?search={cas}", "https://www.ambeed.com/products/search.html?search={cas}"), "Building-block marketplace/catalog; price may be JS/session rendered.", "medium", 86, "marketplace", "public_or_js_mixed"),
    SupplierAdapter("A2B Chem", ("a2bchem.com",), ("https://www.a2bchem.com/search.aspx?search={cas}",), "Building-block supplier; useful candidate source.", "medium", 84, "standard", "public_or_quote_mixed"),
    SupplierAdapter("Enamine", ("enaminestore.com", "enamine.net"), ("https://enaminestore.com/catalogsearch/result/?q={cas}",), "Screening/building block catalog; may need account/API for accurate price.", "medium", 82, "standard", "quote_or_account_mixed"),
    SupplierAdapter("Matrix Scientific", ("matrixscientific.com",), ("https://www.matrixscientific.com/search?query={cas}",), "Specialty chemical supplier; public price varies.", "medium", 80, "standard", "public_or_quote_mixed"),
    SupplierAdapter("Santa Cruz Biotechnology", ("scbt.com",), ("https://www.scbt.com/search?query={cas}",), "Research chemicals/biochemicals; some public prices.", "medium", 78, "standard", "public_or_quote_mixed"),
    SupplierAdapter("CymitQuimica", ("cymitquimica.com",), ("https://cymitquimica.com/products?search={cas}",), "European catalog/marketplace; public price often visible.", "medium", 76, "marketplace", "public_price_mixed"),
    SupplierAdapter("Toronto Research Chemicals", ("trc-canada.com",), ("https://www.trc-canada.com/search?search={cas}",), "Reference/specialty chemicals; public price visibility varies.", "medium", 74, "standard", "public_or_quote_mixed"),

    SupplierAdapter("Fisher Scientific", ("fishersci.com",), ("https://www.fishersci.com/us/en/catalog/search/products?keyword={cas}", "https://www.fishersci.com/us/en/browse/cas/{cas}"), "Large distributor; pricing often account/login-specific.", "low", 60, "login_gated", "login_price_common"),
    SupplierAdapter("Thermo Fisher / Alfa Aesar", ("thermofisher.com", "alfa.com"), ("https://www.thermofisher.com/search/results?keyword={cas}",), "Thermo/Alfa/Acros frequently require account/session JS pricing.", "low", 58, "login_gated", "login_price_common"),
    SupplierAdapter("Sigma-Aldrich", ("sigmaaldrich.com", "milliporesigma.com"), ("https://www.sigmaaldrich.com/US/en/search/{cas}",), "Strong catalog coverage; prices often country/account/session dependent.", "low", 56, "login_gated", "login_price_common"),
    SupplierAdapter("VWR / Avantor", ("vwr.com", "avantorsciences.com"), ("https://us.vwr.com/store/search?keyword={cas}",), "Distributor catalog; account-specific pricing common.", "low", 54, "login_gated", "login_price_common"),

    SupplierAdapter("ChemicalBook", ("chemicalbook.com",), ("https://www.chemicalbook.com/Search_EN.aspx?keyword={cas}",), "Supplier directory; useful RFQ breadth, not price source-of-truth.", "directory", 44, "directory", "directory_rfq"),
    SupplierAdapter("ChemBlink", ("chemblink.com",), ("https://www.chemblink.com/search.aspx?search={cas}",), "Supplier directory; useful for RFQ discovery.", "directory", 42, "directory", "directory_rfq"),
    SupplierAdapter("ChemExper", ("chemexper.com",), ("https://www.chemexper.com/search/cas/{cas}.html",), "Supplier directory; broad catalog leads, not reliable pricing.", "directory", 40, "directory", "directory_rfq"),
    SupplierAdapter("LookChem", ("lookchem.com",), ("https://www.lookchem.com/cas-{cas}.html",), "Supplier directory; useful RFQ lead source.", "directory", 38, "directory", "directory_rfq"),
)

PUBLIC_PRICE_SUPPLIERS = tuple(a.name for a in ADAPTERS if a.source_tier == "price_first")


def sorted_adapters(tiers: tuple[str, ...] | None = None) -> list[SupplierAdapter]:
    adapters = [a for a in ADAPTERS if tiers is None or a.source_tier in tiers]
    return sorted(adapters, key=lambda a: a.search_priority, reverse=True)


def adapter_for_url(url: str) -> SupplierAdapter | None:
    for adapter in ADAPTERS:
        if adapter.matches(url):
            return adapter
    return None


def supplier_name_for_url(url: str, fallback: str = "Unknown supplier") -> str:
    adapter = adapter_for_url(url)
    if adapter:
        return adapter.name
    host = urlparse(url).netloc.lower().replace("www.", "")
    return host.split(".")[0].replace("-", " ").title() if host else fallback


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    path = re.sub(r"/+$", "", unquote(parsed.path or ""))
    # Collapse common language prefixes so GLP Bio /jp /kr /de etc. do not appear as separate products.
    parts = [p for p in path.split("/") if p]
    if parts and parts[0].lower() in LANG_PREFIXES:
        parts = parts[1:]
    path = "/" + "/".join(parts) if parts else ""
    query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in SEARCH_QUERY_KEYS]
    return urlunparse((parsed.scheme or "https", host, path, "", urlencode(query_pairs), ""))


def supplier_key_from_url(url: str) -> str:
    adapter = adapter_for_url(url)
    if adapter:
        return adapter.name
    return supplier_name_for_url(url)


def direct_search_results(cas: str, tier: str | None = None):
    from services.search_service import SearchResult
    out = []
    adapters = sorted_adapters((tier,) if tier else None)
    for adapter in adapters:
        for template in adapter.search_url_templates:
            out.append(SearchResult(
                title=f"{adapter.name} CAS search",
                url=template.format(cas=cas),
                snippet=f"Adapter seed v15. Tier: {adapter.source_tier}. Expected behavior: {adapter.expected_behavior}. {adapter.notes}",
                source=f"adapter_seed_v15_{adapter.source_tier}",
                supplier_hint=adapter.name,
            ))
    return out


def _catalog_token_is_plausible(token: str) -> bool:
    clean = token.strip().strip(".,;:)#/")
    low = clean.lower()
    if low in BAD_CATALOG_TOKENS or len(clean) < 4 or len(clean) > 60:
        return False
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", clean):
        return False
    if re.search(r"(?i)(search|result|login|register|cart|checkout|shipping|happy|customer|html)$", clean):
        return False
    # Avoid accepting normal words from page text. Catalog/SKU values usually contain digits and letters/separators.
    if not re.search(r"\d", clean):
        return False
    if not (re.search(r"[A-Za-z]", clean) or re.search(r"[-_/]", clean)):
        return False
    return True


def extract_catalog_number(*texts: str) -> str | None:
    hay = " | ".join(t for t in texts if t)
    for pattern in CATALOG_PATTERNS:
        for m in pattern.finditer(hay):
            token = m.group(1).strip().strip(".,;:)")
            if _catalog_token_is_plausible(token):
                return token[:60]
    return None


def extract_snippet_price(snippet: str) -> float | None:
    m = PRICE_RE.search(snippet or "")
    if not m:
        return None
    try:
        val = float((m.group(1) or m.group(2)).replace(",", ""))
        # Guard obvious junk values like free samples or pagination counts masquerading as prices.
        return val if val > 0.01 else None
    except Exception:
        return None


def classify_price_visibility(listed_price: float | None, text: str = "", snippet_price: float | None = None, extraction_status: str = "success") -> str:
    hay = text or ""
    if extraction_status.startswith("failed"):
        return PRICE_FAILED
    if listed_price is not None:
        return PRICE_PUBLIC
    if snippet_price is not None:
        return PRICE_SNIPPET
    if re.search(r"(?i)(choose\s+your\s+location|select\s+(?:your\s+)?(?:country|location)|ship\s+to|shipping\s+country|region\s+selection)", hay):
        return PRICE_LOCATION
    if LOGIN_RE.search(hay):
        return PRICE_LOGIN
    if QUOTE_RE.search(hay):
        return PRICE_QUOTE
    return PRICE_HIDDEN


def best_action_for_status(price_visibility_status: str) -> str:
    if price_visibility_status == PRICE_PUBLIC:
        return "Use as catalog price evidence"
    if price_visibility_status == PRICE_SNIPPET:
        return "Open source and verify snippet price"
    if price_visibility_status == PRICE_CANDIDATE:
        return "Review as tentative lead; verify product page"
    if price_visibility_status == PRICE_LOCATION:
        return "Select USA/location on source, then verify price"
    if price_visibility_status == PRICE_LOGIN:
        return "Login/check account price or RFQ"
    if price_visibility_status == PRICE_QUOTE:
        return "Send RFQ"
    if price_visibility_status == PRICE_FAILED:
        return "Open source manually"
    return "Check source / RFQ"
