"""Trade ledger and position engine baseline implementation."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from ai_trading_coach.domain.contracts import LedgerInput, LedgerOutput
from ai_trading_coach.domain.enums import AssetType, TradeSide
from ai_trading_coach.domain.models import (
    PnLSnapshot,
    PositionHolding,
    PositionLot,
    PositionSnapshot,
    TickerPnL,
    TradeLedger,
    TradeOutcomeCandidate,
)


class BasicTradeLedgerPositionEngine:
    """Average-cost ledger engine with degraded behavior on missing prices."""

    def rebuild(self, data: LedgerInput) -> LedgerOutput:
        events = sorted(
            [*data.historical_events, *data.todays_events],
            key=lambda event: (event.trade_date, event.event_id),
        )

        qty_by_ticker: dict[str, float] = defaultdict(float)
        cost_by_ticker: dict[str, float] = defaultdict(float)
        entry_date_by_ticker: dict[str, date] = {}
        realized_by_ticker: dict[str, float] = defaultdict(float)

        outcomes: list[TradeOutcomeCandidate] = []

        for event in events:
            ticker = event.ticker
            qty = qty_by_ticker[ticker]
            total_cost = cost_by_ticker[ticker]

            if event.side == TradeSide.BUY:
                unit_price = event.unit_price or 0.0
                qty_by_ticker[ticker] = qty + event.quantity
                cost_by_ticker[ticker] = total_cost + (unit_price * event.quantity) + event.fees
                entry_date_by_ticker.setdefault(ticker, event.trade_date)
                outcomes.append(
                    TradeOutcomeCandidate(
                        ticker=ticker,
                        direction=event.side,
                        confidence=0.7,
                        summary="Position increased",
                    )
                )
                continue

            sell_price = event.unit_price or 0.0
            if qty <= 0:
                outcomes.append(
                    TradeOutcomeCandidate(
                        ticker=ticker,
                        direction=event.side,
                        confidence=0.3,
                        summary="Sell encountered without open quantity",
                    )
                )
                continue

            effective_qty = min(event.quantity, qty)
            avg_cost = total_cost / qty if qty > 0 else 0.0
            realized = (sell_price - avg_cost) * effective_qty - event.fees

            remaining_qty = qty - effective_qty
            remaining_cost = total_cost - (avg_cost * effective_qty)
            qty_by_ticker[ticker] = max(0.0, remaining_qty)
            cost_by_ticker[ticker] = max(0.0, remaining_cost)
            realized_by_ticker[ticker] += realized

            outcomes.append(
                TradeOutcomeCandidate(
                    ticker=ticker,
                    direction=event.side,
                    confidence=0.8,
                    summary=f"Realized PnL={realized:.2f}",
                )
            )

        open_positions: list[PositionHolding] = []
        closed_positions: list[PositionHolding] = []
        ticker_pnls: list[TickerPnL] = []
        missing_price_tickers: list[str] = []
        total_market_value = 0.0
        total_cost_basis = 0.0
        total_unrealized = 0.0

        for ticker in sorted(set(list(qty_by_ticker.keys()) + list(realized_by_ticker.keys()))):
            qty = qty_by_ticker[ticker]
            cost = cost_by_ticker[ticker]
            realized = realized_by_ticker[ticker]
            avg_cost = (cost / qty) if qty > 0 else 0.0

            market_price = data.latest_prices.get(ticker)
            market_value = (market_price * qty) if (qty > 0 and market_price is not None) else None
            unrealized = None
            if qty > 0 and market_price is not None:
                unrealized = (market_price - avg_cost) * qty
                total_market_value += market_value or 0.0
                total_cost_basis += cost
                total_unrealized += unrealized
            elif qty > 0:
                missing_price_tickers.append(ticker)

            ticker_pnls.append(
                TickerPnL(
                    ticker=ticker,
                    realized_pnl=realized,
                    unrealized_pnl=unrealized,
                )
            )

            if qty > 0:
                entry_date = entry_date_by_ticker.get(ticker, data.run_date)
                open_positions.append(
                    PositionHolding(
                        ticker=ticker,
                        asset_type=AssetType.STOCK,
                        quantity=qty,
                        avg_cost=avg_cost,
                        market_price=market_price,
                        market_value=market_value,
                        unrealized_pnl=unrealized,
                        holding_period_days=max(0, (data.run_date - entry_date).days),
                        lots=[
                            PositionLot(
                                lot_id=f"lot_{ticker}_{entry_date.isoformat()}",
                                ticker=ticker,
                                entry_date=entry_date,
                                quantity_open=qty,
                                cost_basis_per_unit=max(avg_cost, 0.0001),
                            )
                        ],
                    )
                )
            else:
                closed_positions.append(
                    PositionHolding(
                        ticker=ticker,
                        asset_type=AssetType.STOCK,
                        quantity=0,
                        avg_cost=0.0,
                        market_price=market_price,
                        market_value=0.0,
                        unrealized_pnl=0.0,
                        holding_period_days=None,
                    )
                )

        total_realized = sum(item.realized_pnl for item in ticker_pnls)
        pnl_snapshot = PnLSnapshot(
            snapshot_id=f"pnl_{data.user_id}_{data.run_date.isoformat()}",
            user_id=data.user_id,
            as_of_date=data.run_date,
            currency="USD",
            realized_pnl=total_realized,
            unrealized_pnl=(total_unrealized if open_positions else None),
            total_pnl=(total_realized + total_unrealized if open_positions else total_realized),
            by_ticker=ticker_pnls,
            missing_price_tickers=missing_price_tickers,
        )

        position_snapshot = PositionSnapshot(
            snapshot_id=f"pos_{data.user_id}_{data.run_date.isoformat()}",
            user_id=data.user_id,
            as_of_date=data.run_date,
            holdings=open_positions,
            total_market_value=(total_market_value if open_positions else None),
            total_cost_basis=(total_cost_basis if open_positions else None),
            exposure_by_asset={"stock": total_market_value} if total_market_value else {},
        )

        ledger = TradeLedger(
            ledger_id=f"ledger_{data.user_id}_{data.run_date.isoformat()}",
            user_id=data.user_id,
            as_of_date=data.run_date,
            events=events,
            open_positions=open_positions,
            closed_positions=closed_positions,
            missing_price_tickers=missing_price_tickers,
            outcome_candidates=outcomes,
        )

        return LedgerOutput(
            ledger=ledger,
            position_snapshot=position_snapshot,
            pnl_snapshot=pnl_snapshot,
        )


# Backward-compatible alias
PlaceholderTradeLedgerPositionEngine = BasicTradeLedgerPositionEngine
