"""
modules/openai_insights.py

Optional AI-assisted dataset explanation.

It does nothing unless OPENAI_API_KEY is configured.
Uses the current OpenAI Python SDK Responses API.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def generate_dataset_insights(
    df: pd.DataFrame,
    *,
    model: str | None = None,
    max_rows: int = 12,
) -> str:
    """
    Ask an OpenAI model for a concise analyst-style interpretation.

    No API key = clear RuntimeError rather than silently failing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Set it as an environment variable before using AI insights."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Python package is not installed. Run: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key)

    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    numeric = df.select_dtypes(include="number")
    summary: dict[str, Any] = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in df.columns],
        "missing_values": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
            if int(v) > 0
        },
        "numeric_summary": (
            numeric.describe().round(3).head(8).to_dict()
            if not numeric.empty
            else {}
        ),
        "sample": df.head(max_rows).astype(str).to_dict(orient="records"),
    }

    prompt = (
        "You are a data analyst reviewing a dataset. "
        "Based only on the supplied dataset profile, provide: "
        "1) three important observations, "
        "2) two data-quality risks, "
        "3) three useful business/ML questions to investigate. "
        "Do not invent facts that are not supported by the profile. "
        f"Dataset profile:\n{summary}"
    )

    response = client.responses.create(
        model=chosen_model,
        input=prompt,
    )

    return response.output_text
