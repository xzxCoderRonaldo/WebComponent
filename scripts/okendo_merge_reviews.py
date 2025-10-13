#!/usr/bin/env python3
import argparse
import os
import sys
import time
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_BASE_URL = os.getenv("OKENDO_API_BASE", "https://api.okendo.io/merchant")
DEFAULT_PAGE_SIZE = int(os.getenv("OKENDO_PAGE_SIZE", "100"))
DEFAULT_TIMEOUT = float(os.getenv("OKENDO_TIMEOUT", "15"))
RETRY_MAX = int(os.getenv("OKENDO_RETRY_MAX", "3"))
RETRY_BACKOFF = float(os.getenv("OKENDO_RETRY_BACKOFF", "1.5"))


def _build_session(token: str, timeout: float) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    # store default timeout into session object (pattern via wrapper)
    session.request = _wrap_request_with_timeout(session.request, timeout)  # type: ignore
    return session


def _wrap_request_with_timeout(request_fn, timeout: float):
    def wrapped(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return request_fn(method, url, **kwargs)
    return wrapped


def _request_with_retries(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = session.request(method, url, **kwargs)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"Server error {resp.status_code}")
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exc = exc
            if attempt == RETRY_MAX:
                break
            sleep_s = RETRY_BACKOFF ** attempt
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def fetch_reviews(
    session: requests.Session,
    base_url: str,
    page_size: int,
    product_id: Optional[str] = None,
    sku: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    extra_params: Optional[List[Tuple[str, str]]] = None,
) -> Iterable[dict]:
    """Generator yielding all reviews with pagination.

    Notes:
    - The exact parameters depend on Okendo docs. We include flexible pass-through via extra_params.
    - Pagination approach assumes either `page`/`perPage` or `cursor` style. We'll try common `page`.
    """
    page = 1
    while True:
        params: List[Tuple[str, str]] = [("perPage", str(page_size)), ("page", str(page))]
        if product_id:
            params.append(("productId", product_id))
        if sku:
            params.append(("sku", sku))
        if created_after:
            params.append(("createdAfter", created_after))
        if created_before:
            params.append(("createdBefore", created_before))
        if extra_params:
            params.extend(extra_params)

        url = f"{base_url.rstrip('/')}/reviews"
        resp = _request_with_retries(session, "GET", url, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch reviews: {resp.status_code} {resp.text}")

        data = resp.json()
        # Expect shapes like { items: [...], pagination: { page, perPage, totalPages } }
        items = data.get("items") if isinstance(data, dict) else None
        if items is None:
            # Fallback to assuming array
            items = data if isinstance(data, list) else []
        for review in items:
            yield review

        # Pagination detection
        pagination = data.get("pagination") if isinstance(data, dict) else None
        if pagination and "totalPages" in pagination and "page" in pagination:
            total_pages = int(pagination.get("totalPages", 1))
            current_page = int(pagination.get("page", page))
            if current_page >= total_pages:
                break
            page = current_page + 1
        else:
            # If no explicit pagination object, stop after one page
            break


def get_review_variant_key(review: dict, variant_key_fields: List[str]) -> str:
    """Derive the variant key from a review using a list of candidate fields.

    Default order prefers 'variantSku', then 'sku', then 'variantId', then 'variantExternalId'.
    """
    for field in variant_key_fields:
        value = review
        for part in field.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        if isinstance(value, (str, int)) and value is not None:
            return str(value)
    return "__UNKNOWN_VARIANT__"


def group_reviews_by_variant(reviews: Iterable[dict], variant_key_fields: List[str]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for r in reviews:
        key = get_review_variant_key(r, variant_key_fields)
        grouped[key].append(r)
    return grouped


def summarize_group(grouped: Dict[str, List[dict]], top_n: int = 3) -> List[dict]:
    summary: List[dict] = []
    for variant_key, items in grouped.items():
        # Simple fields extraction for preview
        previews = []
        for r in items[:top_n]:
            previews.append({
                "id": r.get("id"),
                "rating": r.get("rating"),
                "title": r.get("title"),
                "body": r.get("body"),
                "createdAt": r.get("createdAt"),
            })
        summary.append({
            "variantKey": variant_key,
            "reviewCount": len(items),
            "sample": previews,
        })
    # sort by reviewCount desc
    summary.sort(key=lambda x: x["reviewCount"], reverse=True)
    return summary


def parse_kv_params(kvs: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in kvs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--param expects key=value, got: {item}"
            )
        k, v = item.split("=", 1)
        out.append((k, v))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Okendo reviews and group by variant for combined display.",
    )
    parser.add_argument("--token", default=os.getenv("OKENDO_API_TOKEN"), help="Okendo API token (or env OKENDO_API_TOKEN)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Base URL (default {DEFAULT_BASE_URL})")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Page size for pagination")
    parser.add_argument("--product-id", help="Filter by productId (per Okendo docs)")
    parser.add_argument("--sku", help="Filter by sku/variantSku if supported")
    parser.add_argument("--created-after", help="ISO datetime filter if supported")
    parser.add_argument("--created-before", help="ISO datetime filter if supported")
    parser.add_argument("--param", action="append", default=[], help="Extra query param as key=value; can repeat")
    parser.add_argument("--variant-key-fields", default="variantSku,sku,variantId,variantExternalId,productVariant.id,productVariant.sku", help="Comma-separated candidate fields for variant key lookup")
    parser.add_argument("--output", choices=["summary", "json"], default="summary", help="Output mode")
    parser.add_argument("--top-n", type=int, default=3, help="Sample size per variant in summary output")

    args = parser.parse_args(argv)

    if not args.token:
        print("Missing API token. Provide --token or env OKENDO_API_TOKEN", file=sys.stderr)
        return 2

    # Build session
    session = _build_session(args.token, DEFAULT_TIMEOUT)

    # Extra params
    extra_params = parse_kv_params(args.param)

    reviews = list(fetch_reviews(
        session=session,
        base_url=args.base_url,
        page_size=args.page_size,
        product_id=args.product_id,
        sku=args.sku,
        created_after=args.created_after,
        created_before=args.created_before,
        extra_params=extra_params,
    ))

    variant_key_fields = [f.strip() for f in args.variant_key_fields.split(',') if f.strip()]
    grouped = group_reviews_by_variant(reviews, variant_key_fields)

    if args.output == "json":
        print(json.dumps(grouped, ensure_ascii=False, indent=2))
        return 0

    # summary output
    summary = summarize_group(grouped, top_n=args.top_n)
    print(json.dumps({
        "totalReviews": len(reviews),
        "variantGroups": summary,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
