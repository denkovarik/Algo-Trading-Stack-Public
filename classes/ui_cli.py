# ui_cli.py

from classes.ui_base import UserInterface
from classes.ui_helpers import format_portfolio

class CLIUserInterface(UserInterface):
    def display_message(self, message: str):
        print(message)

    def get_user_input(self, prompt: str) -> str:
        return input(prompt)

    def display_portfolio(self, portfolio):
        print(self.format_portfolio(portfolio))

    def display_order_status(self, order_id, status):
        print(f"Order {order_id} status: {status}")

    def format_portfolio(self, portfolio):
        lines = []
        lines.append(f"Cash: ${portfolio.get('cash', 0):,.2f}")
        lines.append(f"Total Equity: ${portfolio.get('total_equity', 0):,.2f}\n")
        lines.append("Positions:")
        positions = portfolio.get("positions", {})
        if not positions:
            lines.append("  (none)")
        else:
            for sym, pos in positions.items():
                lines.append(f"  {sym}: qty={pos['qty']} avg_entry={pos['avg_entry_price']:.2f} "
                    f"realized={pos['realized_pnl']:.2f} unrealized={pos['unrealized_pnl']:.2f}")
        lines.append("\nOpen Orders:")
        orders = portfolio.get("open_orders", [])
        if not orders:
            lines.append("  (none)")
        else:
            for o in orders:
                lines.append(f"  #{o['order_id']} {o['side']} {o['qty']} {o['symbol']} " 
                    + "@ {o['price']} status={o['status']}")
        return "\n".join(lines)
        
