def market_representation_status() -> dict[str, object]:
    return {
        "implemented": ["kmeans_submarket_representation"],
        "future_roadmap": [
            "boundary-aware neighborhood encoders",
            "multi-scale spatial context",
            "richer submarket persistence logic",
        ],
    }
