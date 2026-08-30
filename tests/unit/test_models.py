from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.ingestion.models import (
    OrderBookEntry,
    OrderBookSnapshot,
    PriceSnapshot,
    VolumeSnapshot,
)


def test_price_snapshot_valid() -> None:
    now = datetime.now(UTC)
    snapshot = PriceSnapshot(
        timestamp=now,
        symbol="btc",
        price=65000.5,
        currency="USD",
        source="coingecko",
    )
    assert snapshot.symbol == "BTC"
    assert snapshot.price == 65000.5
    assert snapshot.currency == "USD"
    assert snapshot.source == "coingecko"
    assert snapshot.timestamp == now


def test_price_snapshot_invalid_price() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        PriceSnapshot(timestamp=now, symbol="BTC", price=0.0)

    with pytest.raises(ValidationError):
        PriceSnapshot(timestamp=now, symbol="BTC", price=-10.0)


def test_price_snapshot_strict_types() -> None:
    now = datetime.now(UTC)
    # String for price should fail in strict mode
    with pytest.raises(ValidationError):
        PriceSnapshot(timestamp=now, symbol="BTC", price="65000.0")  # type: ignore[arg-type]


def test_price_snapshot_forbid_extra() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        PriceSnapshot(
            timestamp=now,
            symbol="BTC",
            price=100.0,
            extra_field="invalid",  # type: ignore[call-arg]
        )


def test_volume_snapshot_valid() -> None:
    now = datetime.now(UTC)
    volume_snap = VolumeSnapshot(
        timestamp=now,
        symbol="eth",
        volume=0.0,
        currency="USD",
    )
    assert volume_snap.symbol == "ETH"
    assert volume_snap.volume == 0.0

    pos_volume_snap = VolumeSnapshot(
        timestamp=now,
        symbol="ETH",
        volume=123456.78,
    )
    assert pos_volume_snap.volume == 123456.78


def test_volume_snapshot_invalid_volume() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        VolumeSnapshot(timestamp=now, symbol="ETH", volume=-1.0)


def test_order_book_entry_valid() -> None:
    entry = OrderBookEntry(price=100.0, amount=2.5)
    assert entry.price == 100.0
    assert entry.amount == 2.5


def test_order_book_entry_invalid() -> None:
    with pytest.raises(ValidationError):
        OrderBookEntry(price=0.0, amount=1.0)

    with pytest.raises(ValidationError):
        OrderBookEntry(price=100.0, amount=-0.5)


def test_order_book_snapshot_valid() -> None:
    now = datetime.now(UTC)
    snapshot = OrderBookSnapshot(
        timestamp=now,
        symbol="sol",
        bids=[OrderBookEntry(price=150.0, amount=10.0)],
        asks=[OrderBookEntry(price=151.0, amount=5.0)],
        source="binance",
    )
    assert snapshot.symbol == "SOL"
    assert len(snapshot.bids) == 1
    assert len(snapshot.asks) == 1
    assert snapshot.bids[0].price == 150.0
    assert snapshot.asks[0].price == 151.0


def test_order_book_snapshot_invalid_entry() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        OrderBookSnapshot(
            timestamp=now,
            symbol="SOL",
            bids=[{"price": -10.0, "amount": 5.0}],  # type: ignore[list-item]
        )
