from dc_reif.market_representation import market_representation_status


def test_market_representation_status_marks_future_extensions():
    status = market_representation_status()
    assert "kmeans_submarket_representation" in status["implemented"]
    assert status["future_roadmap"]
