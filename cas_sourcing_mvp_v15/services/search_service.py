from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse
import re
import requests
from bs4 import BeautifulSoup

from services.supplier_adapters import ADAPTERS, PUBLIC_PRICE_SUPPLIERS, direct_search_results, supplier_name_for_url, canonicalize_url

DEFAULT_SUPPLIER_DOMAINS = [domain for adapter in ADAPTERS for domain in adapter.domains]
SUPPLIER_NAME_HINTS = {domain: adapter.name for adapter in ADAPTERS for domain in adapter.domains}


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    supplier_hint: str = ""


def supplier_hint_from_url(url: str) -> str:
    return supplier_name_for_url(url)


def build_cas_supplier_queries(cas_number: str, chemical_name: str | None = None) -> list[str]:
    cas = cas_number.strip()
    chem = (chemical_name or "").strip()
    price_first_sites = " OR ".join(PUBLIC_PRICE_SUPPLIERS[:8])
    base_terms = [
        f'"{cas}" "Size" "Price" "Stock"',
        f'"{cas}" "Pack Size" "Price"',
        f'"{cas}" "USD" "In stock"',
        f'"{cas}" "Bulk Inquiry" "Price"',
        f'"{cas}" "catalog no" "price"',
        f'"{cas}" ({price_first_sites})',
        f'"{cas}" supplier price',
        f'"{cas}" catalog price',
        f'"{cas}" buy chemical',
        f'"{cas}" quote',
    ]
    if chem:
        base_terms.extend([
            f'"{cas}" "{chem}" "Size" "Price"',
            f'"{chem}" "{cas}" price',
            f'"{chem}" "{cas}" "pack size"',
        ])
    return base_terms


def direct_supplier_search_urls(cas_number: str, tier: str | None = None) -> list[SearchResult]:
    return direct_search_results(cas_number.strip(), tier=tier)


def serpapi_search(
    queries: Iterable[str],
    api_key: str,
    max_results_per_query: int = 8,
    timeout: int = 20,
) -> list[SearchResult]:
    if not api_key:
        return []
    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    endpoint = "https://serpapi.com/search.json"
    for query in queries:
        params = {"engine": "google", "q": query, "api_key": api_key, "num": max_results_per_query}
        try:
            response = requests.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        for item in payload.get("organic_results", [])[:max_results_per_query]:
            url = item.get("link") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(SearchResult(
                title=item.get("title") or "Untitled search result",
                url=url,
                snippet=item.get("snippet") or "",
                source="serpapi",
                supplier_hint=supplier_hint_from_url(url),
            ))
    return results


def filter_likely_supplier_results(results: list[SearchResult]) -> list[SearchResult]:
    filtered: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        if result.url in seen:
            continue
        seen.add(result.url)
        haystack = f"{result.title} {result.url} {result.snippet}".lower()
        if any(domain in haystack for domain in DEFAULT_SUPPLIER_DOMAINS):
            filtered.append(result)
            continue
        if any(term in haystack for term in ["supplier", "price", "quote", "buy", "catalog", "chemical", "cas"]):
            filtered.append(result)
    return filtered


_PRODUCT_HINT_RE = re.compile(r"(product|catalog|item|sku|compound|chemical|shop|store|/p/|/pd/|details|order|cart)", re.I)
_BAD_LINK_RE = re.compile(
    r"(privacy|terms|basket|login|signin|register|contact|about|careers|linkedin|facebook|twitter|youtube|instagram|cookie|pdf|orders$|order-status|quick-order|promotions|sustainable|all-product-categories|clear-all-filters|clear filters)",
    re.I,
)
_GENERIC_LINK_TEXT_RE = re.compile(r"(?i)^(home|products?|menu|cart|login|sign in|register|contact|about|view|details|more|buy now|add to cart|clear all filters|search)$")


def _same_domain(url_a: str, url_b: str) -> bool:
    try:
        a = urlparse(url_a).netloc.replace("www.", "")
        b = urlparse(url_b).netloc.replace("www.", "")
        return a and b and (a == b or a.endswith("." + b) or b.endswith("." + a))
    except Exception:
        return False


def _clean_short(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def _node_context(a_tag, limit: int = 1500) -> str:
    contexts = []
    for parent in [a_tag, a_tag.parent, a_tag.find_parent("li"), a_tag.find_parent("tr"), a_tag.find_parent("div")]:
        if parent is None:
            continue
        txt = parent.get_text(" ", strip=True)
        if txt and txt not in contexts:
            contexts.append(txt)
    return _clean_short(" | ".join(contexts), limit)


def _seed_is_cas_search(seed_url: str, cas_number: str) -> bool:
    hay = seed_url.lower()
    return cas_number.lower() in hay and any(x in hay for x in ["/cas/", "/search", "catalogsearch", "keyword=", "search=", "q=", "query=", "text="])


def _supplier_product_link_bonus(supplier: str, href: str, text: str, context: str, cas_number: str, seed_url: str) -> int:
    supplier_l = (supplier or "").lower()
    path = urlparse(href).path.lower()
    link_text = _clean_short(text).lower()
    hay = f"{href} {text} {context}".lower()
    if _BAD_LINK_RE.search(hay) or _GENERIC_LINK_TEXT_RE.match(link_text):
        return -100
    if not _seed_is_cas_search(seed_url, cas_number):
        return 0
    product_url = False
    if "medchemexpress" in supplier_l:
        product_url = bool(re.search(r"/[a-z0-9][a-z0-9\-]+\.html$", path) and not path.startswith("/cas/") and "search" not in path)
    elif "adooq" in supplier_l:
        product_url = bool(re.search(r"/[a-z0-9][a-z0-9\-]+\.html$", path) and "catalogsearch" not in path and "search" not in path)
    elif "abmole" in supplier_l:
        product_url = bool(re.search(r"/[a-z0-9][a-z0-9\-]+\.html$", path) and "search" not in path)
    elif "ambeed" in supplier_l:
        product_url = bool(re.search(r"/products/[a-z0-9][a-z0-9\-]+\.html$", path))
    elif "biosynth" in supplier_l:
        product_url = bool("/product" in path or re.search(r"/[a-z0-9][a-z0-9\-]+\.html$", path))
    elif "invivochem" in supplier_l:
        product_url = bool("product" in path or re.search(r"/[a-z0-9][a-z0-9\-]+\.html$", path))
    else:
        product_url = bool(_PRODUCT_HINT_RE.search(path) or re.search(r"/[a-z0-9][a-z0-9\-]+\.html$", path))
    if not product_url:
        return 0
    bonus = 80
    if cas_number.lower() in hay:
        bonus += 40
    if any(tok in hay for tok in ["price", "stock", "size", "pack", "cas", "$", "purity", "catalog"]):
        bonus += 15
    return bonus


def _link_score(href: str, text: str, context: str, cas_number: str, supplier: str = "", seed_url: str = "") -> int:
    hay = f"{href} {text} {context}".lower()
    score = 0
    if cas_number.lower() in hay:
        score += 70
    if _PRODUCT_HINT_RE.search(hay):
        score += 15
    if any(term in hay for term in ["price", "pricing", "$", "pack", "size", "purity", "assay", "cas"]):
        score += 10
    if _BAD_LINK_RE.search(hay):
        score -= 100
    if len(text.strip()) < 3 and cas_number.lower() not in href.lower():
        score -= 20
    score += _supplier_product_link_bonus(supplier, href, text, context, cas_number, seed_url)
    return score


def _browser_headers_and_cookies(url: str) -> tuple[dict[str, str], dict[str, str]]:
    host = urlparse(url).netloc.lower().replace("www.", "")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }
    cookies = {
        "currency": "USD", "Currency": "USD", "currencyCode": "USD", "country": "US",
        "Country": "US", "countryCode": "US", "country_code": "US", "shipping_country": "US",
        "selected_country": "US", "shipToCountry": "US", "defaultCountry": "US", "locale": "en_US",
        "language": "en", "selectedCurrency": "USD", "selectedRegion": "United States",
    }
    if host:
        headers["Referer"] = f"https://{host}/"
    return headers, cookies


def discover_product_links_from_page(result: SearchResult, cas_number: str, timeout: int = 12, max_links: int = 8) -> list[SearchResult]:
    headers, cookies = _browser_headers_and_cookies(result.url)
    try:
        resp = requests.get(result.url, headers=headers, cookies=cookies, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    supplier = result.supplier_hint or supplier_hint_from_url(result.url)
    candidates: list[tuple[int, SearchResult]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(resp.url, a.get("href", ""))
        canon = canonicalize_url(href)
        if not href.startswith("http") or canon in seen:
            continue
        if not _same_domain(resp.url, href):
            continue
        text = _clean_short(a.get_text(" ", strip=True))
        context = _node_context(a)
        score = _link_score(href, text, context, cas_number, supplier=supplier, seed_url=resp.url)
        if score < 70:
            continue
        seen.add(canon)
        candidates.append((score, SearchResult(
            title=text or result.title,
            url=href,
            snippet=f"Expanded from {result.url}. Context: {context[:500]}",
            source="expanded_product_link_v15",
            supplier_hint=supplier,
        )))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in candidates[:max_links]]
