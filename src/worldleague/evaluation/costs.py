"""Attribute per-attempt USD costs with worldleague analyze [--tariffs rates.json]."""

import math


def reported_charge(usage):
    try:
        raw = next((usage.get(k) for k in ("cost", "gateway_cost", "cost_usd") if usage.get(k) is not None), None)
        value = float(raw)
    except (KeyError, ValueError, TypeError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def estimated_charge(usage, tariff):
    """Apply the explicitly selected historical tariff to one request's token usage."""
    if not tariff:
        return None
    inputs = usage.get("prompt_tokens", usage.get("input_tokens"))
    outputs = usage.get("completion_tokens", usage.get("output_tokens"))
    if inputs is None or outputs is None:
        return None
    rates = tariff["rates"]
    if inputs > tariff.get("long_input_threshold", math.inf):
        rates = tariff["long_rates"]
    if not any(key in rates for key in ("cached_input", "write_30m", "write_12h")):
        return (inputs * rates["input"] + outputs * rates["output"]) / 1e6
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if "cached_input" in rates and "cached_tokens" not in details:
        return None
    cached = details.get("cached_tokens", 0)
    writes = details.get("cache_write_tokens", 0)
    ttl = details.get("cache_write_tokens_details") or {}
    long_writes = ttl.get("ttl_12h_tokens", 0)
    short_writes = ttl.get("ttl_30m_tokens", writes - long_writes)
    regular = inputs - cached - writes
    if min(regular, cached, short_writes, long_writes, outputs) < 0 or short_writes + long_writes != writes:
        return None
    return (regular * rates["input"] + cached * rates.get("cached_input", rates["input"])
            + short_writes * rates.get("write_30m", rates["input"])
            + long_writes * rates.get("write_12h", rates["input"]) + outputs * rates["output"]) / 1e6


def cost_totals(rows, tariff):
    reported, estimated, listed = [], [], []
    for row in rows:
        usage = row.get("usage") or {}
        quote = estimated_charge(usage, tariff)
        if quote is not None:
            listed.append(quote)
        charge = reported_charge(usage)
        if charge is not None:
            reported.append(charge)
        elif quote is not None:
            estimated.append(quote)
    known = reported + estimated
    return {
        "tariff_cost_usd": sum(listed) if listed and len(listed) == len(rows) else None,
        "tariff_priced_calls": len(listed),
        "cost_usd": sum(known) if known and len(known) == len(rows) else None,
        "cost_known_usd": sum(known) if known else None,
        "cost_reported_usd": sum(reported) if reported else None,
        "cost_estimated_usd": sum(estimated) if estimated else None,
        "cost_reported_calls": len(reported), "cost_estimated_calls": len(estimated),
        "cost_usd_known_calls": len(known), "cost_unknown_calls": len(rows) - len(known),
    }
