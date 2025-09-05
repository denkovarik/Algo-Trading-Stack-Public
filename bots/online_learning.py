# bots/online_learning.py
from typing import Any

def attach_online_learning(bot: Any, verbose: bool = True) -> None:
    """
    Wrap bot.on_bar so that, after each bar, we allow the exit strategy
    to ingest new fills and update its bandit (if supported).

    Safe to call multiple times (idempotent).
    """
    if not hasattr(bot, "on_bar"):
        return

    # Prevent double-wrapping
    if getattr(bot, "_ol_wrapped", False):
        return

    _orig_on_bar = bot.on_bar

    def _on_bar_with_online(env, *args, **kwargs):
        result = _orig_on_bar(env, *args, **kwargs)
        exit_strategy = getattr(bot, "exit_strategy", None)
        if exit_strategy and hasattr(exit_strategy, "ingest_trade_log"):
            exit_strategy.ingest_trade_log(env)
        return result

    bot.on_bar = _on_bar_with_online
    bot._ol_wrapped = True  # mark as wrapped

    if verbose:
        print("Using online learning")

