# app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    monthly_budget = Column(Float, default=0.0)

    expenses = relationship("Expense", back_populates="user")
    budgets = relationship("Budget", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")


class Expense(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    description = Column(String)
    date = Column(DateTime, server_default=func.now())
    merchant = Column(String)

    user = relationship("User", back_populates="expenses")


class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    keywords = Column(String)  # comma-separated


class Budget(Base):
    __tablename__ = "budgets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    category = Column(String)
    monthly_limit = Column(Float)

    user = relationship("User", back_populates="budgets")


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(String)
    response = Column(String)
    timestamp = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="chat_history")
#new
class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    category = Column(String)
    target_amount = Column(Float)
    period = Column(String)  # "monthly" / "weekly"

    user = relationship("User")
