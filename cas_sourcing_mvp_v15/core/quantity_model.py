from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class SupplierQuantityModel:
    supplier: str
    product_url: str
    product_title: str
    product_form: str
    purity: str
    price_points_used: int
    min_pack_g: float
    max_pack_g: float
    pack_span_x: float
    observed_best_unit_price_per_g: float
    observed_total_price_exponent: float
    literature_unit_discount_gamma: float
    literature_total_price_exponent: float
    effective_total_price_exponent: float
    aggressive_total_price_exponent: float
    conservative_total_price_exponent: float
    curve_r2: float | None
    curve_quality: str
    desired_qty_g: float
    pricing_basis: str
    exact_public_catalog_match: bool
    exact_catalog_pack_g: float | None
    exact_catalog_price_usd: float | None
    interpolation_lower_pack_g: float | None
    interpolation_lower_price_usd: float | None
    interpolation_upper_pack_g: float | None
    interpolation_upper_price_usd: float | None
    nearest_public_pack_g: float
    nearest_public_pack_price_usd: float
    scale_gap_x: float
    catalog_supported_qty_g: float
    reasonable_model_qty_g: float
    reasonable_model_total_usd: float
    target_estimated_total_usd: float
    target_estimated_unit_price_per_g: float
    conservative_total_usd: float
    aggressive_total_usd: float
    estimate_range_low_usd: float
    estimate_range_high_usd: float
    negotiation_anchor_usd: float
    retail_catalog_ceiling_usd: float
    best_catalog_pack_g: float
    best_catalog_pack_price_usd: float
    catalog_pack_count_for_desired: int
    catalog_multiple_cost_usd: float
    suggested_rfq_tiers: str
    product_complexity_class: str
    extrapolation_risk: str
    model_confidence: str
    quantity_decision: str
    model_reason: str


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        if math.isfinite(f) and f > 0:
            return f
    except Exception:
        return None
    return None


def _format_qty_g(qty_g: float) -> str:
    qty_g = float(qty_g)
    if qty_g >= 1000:
        return f"{qty_g / 1000:g} kg"
    if qty_g >= 1:
        return f"{qty_g:g} g"
    if qty_g >= 0.001:
        return f"{qty_g * 1000:g} mg"
    return f"{qty_g * 1_000_000:g} µg"


def _money(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "not available"
    return f"${float(value):,.2f}"


def _nice_qty_g(qty_g: float) -> float:
    if qty_g <= 0 or not math.isfinite(float(qty_g)):
        return 0.0
    q = float(qty_g)
    if q < 1:
        bases = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    elif q < 1000:
        bases = [1, 2.5, 5, 10, 25, 50, 100, 250, 500]
    else:
        bases = [1000, 2500, 5000, 10000, 25000, 50000]
    return min(bases, key=lambda x: abs(math.log(max(x, 1e-12)) - math.log(q)))


def _fit_total_price_curve(points: list[tuple[float, float]]) -> tuple[float, float, float | None, str]:
    if len(points) < 2:
        q, price = points[0]
        alpha = 0.62
        return math.log(price) - alpha * math.log(q), alpha, None, "single-point specialty prior"
    xs = [math.log(q) for q, _ in points]
    ys = [math.log(p) for _, p in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x <= 1e-12:
        q, price = points[-1]
        alpha = 0.62
        return math.log(price) - alpha * math.log(q), alpha, None, "flat-pack specialty prior"
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    alpha = cov / var_x
    alpha = max(0.18, min(0.98, alpha))
    intercept = y_mean - alpha * x_mean
    yhat = [intercept + alpha * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    quality = "observed supplier total-price curve"
    return intercept, alpha, max(0.0, min(1.0, r2)), quality


def _predict_from_anchor(anchor_qty: float, anchor_price: float, target_qty: float, alpha: float) -> float:
    target_qty = max(target_qty, 1e-12)
    anchor_qty = max(anchor_qty, 1e-12)
    return float(anchor_price) * math.pow(target_qty / anchor_qty, alpha)


def _build_points(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Return de-duplicated catalog price points. For duplicate pack sizes, keep the lowest visible public price."""
    by_qty: dict[float, float] = {}
    for _, row in df.iterrows():
        q = _safe_float(row.get("pack_size_g"))
        price = _safe_float(row.get("listed_price_usd"))
        if q is not None and price is not None:
            key = round(q, 12)
            by_qty[key] = min(by_qty.get(key, float("inf")), round(price, 6))
    return sorted((q, p) for q, p in by_qty.items() if math.isfinite(p))


def _nearest_point(points: list[tuple[float, float]], desired: float) -> tuple[float, float]:
    return min(points, key=lambda qp: abs(math.log(qp[0]) - math.log(max(desired, 1e-12))))


def _catalog_resolution(points: list[tuple[float, float]], desired: float) -> dict | None:
    """Resolve exact catalog match or within-ladder interpolation before any scale-up model is used."""
    if not points:
        return None
    pts = sorted(points)
    nearest_q, nearest_p = _nearest_point(pts, desired)

    # Exact public catalog match wins. This prevents the model from estimating a listed pack.
    for q, p in pts:
        if math.isclose(q, desired, rel_tol=1e-6, abs_tol=1e-12):
            return {
                "basis": "exact_public_catalog_match",
                "decision": "Exact public catalog match",
                "target": p,
                "low": p,
                "high": p,
                "reasonable_qty": desired,
                "reasonable_total": p,
                "risk": "low",
                "confidence": "HIGH",
                "exact": True,
                "exact_q": q,
                "exact_p": p,
                "lower_q": q,
                "lower_p": p,
                "upper_q": q,
                "upper_p": p,
                "nearest_q": q,
                "nearest_p": p,
                "reason": (
                    f"Requested quantity exactly matches a public catalog row: {_format_qty_g(q)} at {_money(p)}. "
                    "v15 uses the listed catalog price directly; no interpolation or bulk scale-up was applied."
                ),
            }

    min_q, min_p = pts[0]
    max_q, max_p = pts[-1]

    if desired < min_q:
        return {
            "basis": "minimum_public_pack_floor",
            "decision": "Below public catalog minimum",
            "target": min_p,
            "low": min_p,
            "high": min_p,
            "reasonable_qty": min_q,
            "reasonable_total": min_p,
            "risk": "low",
            "confidence": "MEDIUM",
            "exact": False,
            "exact_q": None,
            "exact_p": None,
            "lower_q": None,
            "lower_p": None,
            "upper_q": min_q,
            "upper_p": min_p,
            "nearest_q": nearest_q,
            "nearest_p": nearest_p,
            "reason": (
                f"Requested quantity is below the smallest public pack. The smallest sellable public pack is "
                f"{_format_qty_g(min_q)} at {_money(min_p)}. v15 reports that pack as the practical catalog floor."
            ),
        }

    if desired < max_q:
        lower = None
        upper = None
        for q, p in pts:
            if q < desired:
                lower = (q, p)
            elif q > desired and upper is None:
                upper = (q, p)
                break
        if lower and upper:
            lq, lp = lower
            uq, up = upper
            # Log-log interpolation is appropriate for price ladders because pack pricing is multiplicative.
            t = (math.log(desired) - math.log(lq)) / (math.log(uq) - math.log(lq))
            target = math.exp(math.log(lp) + t * (math.log(up) - math.log(lp)))
            low = target * 0.94
            high = target * 1.06
            return {
                "basis": "nearest_pack_interpolation",
                "decision": "Nearest-pack interpolation",
                "target": target,
                "low": low,
                "high": high,
                "reasonable_qty": desired,
                "reasonable_total": target,
                "risk": "low",
                "confidence": "HIGH" if len(points) >= 3 else "MEDIUM",
                "exact": False,
                "exact_q": None,
                "exact_p": None,
                "lower_q": lq,
                "lower_p": lp,
                "upper_q": uq,
                "upper_p": up,
                "nearest_q": nearest_q,
                "nearest_p": nearest_p,
                "reason": (
                    f"Requested quantity sits inside the public catalog ladder. v15 interpolated between the nearest listed packs: "
                    f"{_format_qty_g(lq)} at {_money(lp)} and {_format_qty_g(uq)} at {_money(up)}. "
                    f"Estimated catalog-equivalent target: {_money(target)}."
                ),
            }
    return None


def _text_blob(g: pd.DataFrame) -> str:
    cols = [c for c in ["page_title", "chemical_name", "raw_matches", "notes", "supplier", "product_url"] if c in g.columns]
    parts = []
    for c in cols:
        parts.extend(str(x) for x in g[c].dropna().astype(str).head(6).tolist())
    return " ".join(parts).lower()


def _product_complexity(g: pd.DataFrame, max_pack: float, desired: float) -> tuple[str, int]:
    hay = _text_blob(g)
    score = 0
    if max_pack < 0.1:
        score += 2
    elif max_pack < 1:
        score += 1
    if desired / max(max_pack, 1e-12) >= 1000:
        score += 2
    elif desired / max(max_pack, 1e-12) >= 100:
        score += 1
    if any(token in hay for token in [
        "inhibitor", "p450", "ampk", "virus", "glycoside", "natural product", "pharma",
        "bioactive", "medchem", "lignan", "standard", "reference", "assay", "cell",
    ]):
        score += 2
    purity_vals = []
    if "purity_value_pct" in g.columns:
        for value in g["purity_value_pct"].dropna().tolist():
            f = _safe_float(value)
            if f is not None:
                purity_vals.append(f)
    if purity_vals and max(purity_vals) >= 98:
        score += 1
    if len(g) <= 3:
        score += 1
    if score >= 6:
        return "high-complexity specialty chemical", score
    if score >= 3:
        return "specialty organic / medchem-like", score
    return "catalog chemical / lower complexity", score


def _literature_prior_total_exponent(product_class: str) -> tuple[float, float]:
    # LBNL-style quantity scaling: unit price generally falls as purchase quantity increases.
    # Total price exponent = 1 + unit-price discount factor.
    if "lower complexity" in product_class:
        gamma = -0.56
    else:
        gamma = -0.67
    return gamma, 1.0 + gamma


def _support_weight(n: int, span: float, r2: float | None) -> float:
    weight = 0.35
    if n >= 3:
        weight += 0.15
    if n >= 5:
        weight += 0.15
    if span >= 25:
        weight += 0.10
    if span >= 75:
        weight += 0.05
    if r2 is not None and r2 >= 0.95:
        weight += 0.10
    elif r2 is not None and r2 < 0.85:
        weight -= 0.10
    return max(0.25, min(0.85, weight))


def _effective_alpha(alpha_obs: float, alpha_prior: float, product_class: str, scale_gap: float, n: int, span: float, r2: float | None) -> tuple[float, float, float, str, str]:
    w = _support_weight(n, span, r2)
    if "high-complexity" in product_class:
        specialty_floor = 0.56
    elif "specialty" in product_class:
        specialty_floor = 0.48
    else:
        specialty_floor = 0.40
    blended = w * alpha_obs + (1.0 - w) * max(alpha_prior, specialty_floor)
    if scale_gap > 10:
        blended += min(0.12, 0.025 * math.log10(scale_gap / 10.0))
    base = max(specialty_floor, min(0.82, blended))
    uncertainty = 0.07
    if n < 3:
        uncertainty += 0.08
    if r2 is None or r2 < 0.90:
        uncertainty += 0.05
    if scale_gap > 100:
        uncertainty += min(0.12, 0.025 * math.log10(scale_gap / 100.0))
    if "high-complexity" in product_class:
        uncertainty += 0.04
    aggressive = max(0.25, base - uncertainty)
    conservative = min(0.92, base + uncertainty)
    if scale_gap <= 5:
        risk = "low"
    elif scale_gap <= 100:
        risk = "moderate"
    elif scale_gap <= 1000:
        risk = "high"
    else:
        risk = "very high"
    if risk == "very high" or n < 2:
        confidence = "LOW"
    elif risk == "high" or (r2 is not None and r2 < 0.9):
        confidence = "MEDIUM-LOW"
    elif n >= 4 and r2 is not None and r2 >= 0.95:
        confidence = "MEDIUM"
    else:
        confidence = "MEDIUM-LOW"
    return base, aggressive, conservative, risk, confidence


def _catalog_supported_qty(n: int, span: float, max_pack: float, r2: float | None) -> float:
    if n >= 5 and span >= 50 and (r2 is None or r2 >= 0.95):
        multiplier = min(100.0, max(20.0, math.sqrt(span) * 8.0))
    elif n >= 3 and span >= 10:
        multiplier = min(50.0, max(8.0, math.sqrt(span) * 4.0))
    elif n >= 2:
        multiplier = 10.0
    else:
        multiplier = 3.0
    return max_pack * multiplier


def _suggest_rfq_tiers(max_pack: float, desired: float, catalog_supported: float) -> str:
    candidates = [max_pack, max_pack * 2, max_pack * 5, max_pack * 10, _nice_qty_g(catalog_supported)]
    for qty in [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]:
        if max_pack < qty < desired:
            candidates.append(qty)
    candidates.append(desired)
    labels: list[str] = []
    for qty in sorted({round(float(q), 9) for q in candidates if q and q > 0}):
        label = _format_qty_g(qty)
        if label not in labels:
            labels.append(label)
    return ", ".join(labels[:16])


def _decision(desired: float, max_pack: float, catalog_supported: float, risk: str) -> str:
    if desired <= max_pack * 1.000001:
        return "Catalog interpolation"
    if desired <= catalog_supported * 1.000001:
        return "Catalog-supported scale-up estimate"
    if risk in {"high", "very high"}:
        return "Bulk RFQ with scale-up estimate"
    return "Bulk estimate, RFQ recommended"


def build_quantity_models(price_df: pd.DataFrame, desired_qty_g: float) -> pd.DataFrame:
    if price_df is None or price_df.empty or desired_qty_g <= 0:
        return pd.DataFrame()
    df = price_df.copy()
    if "bulk_estimate_eligible" in df.columns:
        df = df[df["bulk_estimate_eligible"].fillna(False).astype(bool)]
    else:
        df = df[df.get("listed_price_usd", pd.Series(dtype=float)).notna()]
    df = df[df.get("pack_size_g", pd.Series(dtype=float)).notna()]
    if df.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["supplier", "product_url", "page_title", "product_form", "purity"] if c in df.columns]
    records: list[SupplierQuantityModel] = []

    for group_key, g in df.groupby(group_cols, dropna=False):
        points = _build_points(g)
        if not points:
            continue

        n = len(points)
        min_pack = min(q for q, _ in points)
        max_pack = max(q for q, _ in points)
        span = max_pack / min_pack if min_pack > 0 else 1.0
        best_qty, best_price = min(points, key=lambda qp: qp[1] / qp[0])
        best_unit = best_price / best_qty
        anchor_qty, anchor_price = max(points, key=lambda x: x[0])
        nearest_qty, nearest_price = _nearest_point(points, float(desired_qty_g))
        catalog_pack_count = int(math.ceil(desired_qty_g / best_qty)) if best_qty > 0 else 0
        catalog_multiple_cost = catalog_pack_count * best_price

        intercept, alpha_obs, r2, quality = _fit_total_price_curve(points)
        product_class, _complexity_score = _product_complexity(g, max_pack, desired_qty_g)
        gamma_prior, alpha_prior = _literature_prior_total_exponent(product_class)
        scale_gap = float(desired_qty_g) / max_pack if max_pack > 0 else float("inf")
        alpha_base, alpha_aggressive, alpha_conservative, risk, confidence = _effective_alpha(
            alpha_obs, alpha_prior, product_class, scale_gap, n, span, r2
        )
        catalog_supported_qty = _catalog_supported_qty(n, span, max_pack, r2)
        rfq_tiers = _suggest_rfq_tiers(max_pack, float(desired_qty_g), catalog_supported_qty)

        catalog_solution = _catalog_resolution(points, float(desired_qty_g))
        if catalog_solution is not None:
            pricing_basis = catalog_solution["basis"]
            decision = catalog_solution["decision"]
            target_total = float(catalog_solution["target"])
            aggressive_total = float(catalog_solution["low"])
            conservative_total = float(catalog_solution["high"])
            low = float(catalog_solution["low"])
            high = float(catalog_solution["high"])
            reasonable_qty = float(catalog_solution["reasonable_qty"])
            reasonable_total = float(catalog_solution["reasonable_total"])
            risk = str(catalog_solution["risk"])
            confidence = str(catalog_solution["confidence"])
            negotiation_anchor = target_total
            reason = str(catalog_solution["reason"])
            exact = bool(catalog_solution["exact"])
            exact_q = catalog_solution["exact_q"]
            exact_p = catalog_solution["exact_p"]
            lower_q = catalog_solution["lower_q"]
            lower_p = catalog_solution["lower_p"]
            upper_q = catalog_solution["upper_q"]
            upper_p = catalog_solution["upper_p"]
        else:
            pricing_basis = "rfq_scaleup_model"
            reasonable_qty = min(float(desired_qty_g), catalog_supported_qty)
            reasonable_total = _predict_from_anchor(anchor_qty, anchor_price, reasonable_qty, alpha_base)
            target_total = _predict_from_anchor(anchor_qty, anchor_price, float(desired_qty_g), alpha_base)
            aggressive_total = _predict_from_anchor(anchor_qty, anchor_price, float(desired_qty_g), alpha_aggressive)
            conservative_total = _predict_from_anchor(anchor_qty, anchor_price, float(desired_qty_g), alpha_conservative)
            low = min(aggressive_total, target_total, conservative_total)
            high = max(aggressive_total, target_total, conservative_total)
            negotiation_anchor = target_total * (0.92 if risk in {"high", "very high"} else 0.96)
            decision = _decision(float(desired_qty_g), max_pack, catalog_supported_qty, risk)
            exact = False
            exact_q = exact_p = lower_q = lower_p = upper_q = upper_p = None
            reason = (
                f"Observed catalog ladder covers {_format_qty_g(min_pack)}–{_format_qty_g(max_pack)} with {n} paired price point(s). "
                f"v15 anchors on the largest public pack ({_format_qty_g(anchor_qty)} at {_money(anchor_price)}) and applies a blended "
                f"total-price scale-up exponent of {alpha_base:.2f} (observed curve {alpha_obs:.2f}; literature prior total exponent {alpha_prior:.2f}; "
                f"product class: {product_class}). For {_format_qty_g(float(desired_qty_g))}, the modeled RFQ target is {_money(target_total)}, "
                f"with a buyer-aggressive lower case of {_money(low)} and supplier-conservative upper case of {_money(high)}. "
                f"The nearest public pack is {_format_qty_g(nearest_qty)} at {_money(nearest_price)}; the public-catalog multiple cost is {_money(catalog_multiple_cost)} "
                f"and is treated only as a retail ceiling. Suggested RFQ tiers: {rfq_tiers}."
            )

        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        lookup = dict(zip(group_cols, group_key))

        records.append(SupplierQuantityModel(
            supplier=str(lookup.get("supplier", "Unknown")),
            product_url=str(lookup.get("product_url", "")),
            product_title=str(lookup.get("page_title", "")),
            product_form=str(lookup.get("product_form", "")),
            purity=str(lookup.get("purity", "")),
            price_points_used=n,
            min_pack_g=float(min_pack),
            max_pack_g=float(max_pack),
            pack_span_x=float(span),
            observed_best_unit_price_per_g=float(best_unit),
            observed_total_price_exponent=float(alpha_obs),
            literature_unit_discount_gamma=float(gamma_prior),
            literature_total_price_exponent=float(alpha_prior),
            effective_total_price_exponent=float(alpha_base),
            aggressive_total_price_exponent=float(alpha_aggressive),
            conservative_total_price_exponent=float(alpha_conservative),
            curve_r2=float(r2) if r2 is not None else None,
            curve_quality=quality,
            desired_qty_g=float(desired_qty_g),
            pricing_basis=pricing_basis,
            exact_public_catalog_match=bool(exact),
            exact_catalog_pack_g=float(exact_q) if exact_q is not None else None,
            exact_catalog_price_usd=float(exact_p) if exact_p is not None else None,
            interpolation_lower_pack_g=float(lower_q) if lower_q is not None else None,
            interpolation_lower_price_usd=float(lower_p) if lower_p is not None else None,
            interpolation_upper_pack_g=float(upper_q) if upper_q is not None else None,
            interpolation_upper_price_usd=float(upper_p) if upper_p is not None else None,
            nearest_public_pack_g=float(nearest_qty),
            nearest_public_pack_price_usd=float(nearest_price),
            scale_gap_x=float(scale_gap),
            catalog_supported_qty_g=float(catalog_supported_qty),
            reasonable_model_qty_g=float(reasonable_qty),
            reasonable_model_total_usd=float(reasonable_total),
            target_estimated_total_usd=float(target_total),
            target_estimated_unit_price_per_g=float(target_total / float(desired_qty_g)),
            conservative_total_usd=float(conservative_total),
            aggressive_total_usd=float(aggressive_total),
            estimate_range_low_usd=float(low),
            estimate_range_high_usd=float(high),
            negotiation_anchor_usd=float(negotiation_anchor),
            retail_catalog_ceiling_usd=float(catalog_multiple_cost),
            best_catalog_pack_g=float(best_qty),
            best_catalog_pack_price_usd=float(best_price),
            catalog_pack_count_for_desired=int(catalog_pack_count),
            catalog_multiple_cost_usd=float(catalog_multiple_cost),
            suggested_rfq_tiers=rfq_tiers,
            product_complexity_class=product_class,
            extrapolation_risk=risk,
            model_confidence=confidence,
            quantity_decision=decision,
            model_reason=reason,
        ))

    out = pd.DataFrame([r.__dict__ for r in records])
    if not out.empty:
        decision_rank = {
            "Exact public catalog match": 0,
            "Nearest-pack interpolation": 1,
            "Below public catalog minimum": 2,
            "Catalog interpolation": 3,
            "Catalog-supported scale-up estimate": 4,
            "Bulk estimate, RFQ recommended": 5,
            "Bulk RFQ with scale-up estimate": 6,
        }
        conf_rank = {"HIGH": 0, "MEDIUM": 1, "MEDIUM-LOW": 2, "LOW": 3}
        risk_rank = {"low": 0, "moderate": 1, "high": 2, "very high": 3}
        out["quantity_decision_rank"] = out["quantity_decision"].map(decision_rank).fillna(9)
        out["model_confidence_rank"] = out["model_confidence"].map(conf_rank).fillna(9)
        out["extrapolation_risk_rank"] = out["extrapolation_risk"].map(risk_rank).fillna(9)
        out = out.sort_values(
            ["quantity_decision_rank", "model_confidence_rank", "extrapolation_risk_rank", "target_estimated_total_usd", "max_pack_g", "price_points_used"],
            ascending=[True, True, True, True, False, False],
        )
    return out
