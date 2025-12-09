# ui_base.py

from abc import ABC, abstractmethod

class UserInterface(ABC):
    @abstractmethod
    def display_message(self, message: str):
        pass

    @abstractmethod
    def get_user_input(self, prompt: str) -> str:
        pass

    @abstractmethod
    def display_portfolio(self, portfolio):
        pass

    @abstractmethod
    def display_order_status(self, order_id, status):
        pass

