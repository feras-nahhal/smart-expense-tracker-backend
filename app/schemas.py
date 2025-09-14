# app/schemas.py
from pydantic import BaseModel, Field, EmailStr, constr
from datetime import datetime

# -----------------------------
# User Schemas
# -----------------------------
class UserCreate(BaseModel):
    name: constr(strip_whitespace=True, min_length=1, max_length=50)
    email: EmailStr
    password: constr(min_length=6)
    monthly_budget: float = Field(0.0, ge=0)


class UserLogin(BaseModel):
    email: EmailStr
    password: constr(min_length=6)


# -----------------------------
# Expense Schemas
# -----------------------------
class ExpenseIn(BaseModel):
    user_id: int
    text: str = Field(..., min_length=1)


class ExpenseOut(BaseModel):
    id: int
    user_id: int
    amount: float = Field(..., ge=0)
    category_id: int  # ✅ foreign key now
    description: str | None = None
    date: datetime
    merchant: str | None = None

    class Config:
        from_attributes = True  # ✅ Pydantic v2 fix


class ExpenseQuick(BaseModel):
    user_id: int
    amount: float = Field(..., gt=0, lt=100000)  # ✅ between 0 and 100k
    category: str   # ✅ accept plain category name
    description: str | None = None
    merchant: str | None = None


# -----------------------------
# Goal Schemas
# -----------------------------
class GoalCreate(BaseModel):
    user_id: int
    category: str   # ✅ accept plain text name
    target_amount: float = Field(..., ge=0)
    period: constr(pattern="^(monthly|weekly)$")  # ✅ only monthly/weekly allowed


class GoalOut(GoalCreate):
    id: int

    class Config:
        from_attributes = True  # ✅ Pydantic v2 fix


# -----------------------------
# Category Schemas
# -----------------------------
class CategoryBase(BaseModel):
    name: str
    keywords: str | None = None


class CategoryCreate(CategoryBase):
    pass


class CategoryOut(CategoryBase):
    id: int

    class Config:
        orm_mode = True


# -----------------------------
# ChatHistory Schemas
# -----------------------------

class ChatHistoryBase(BaseModel):
    user_id: int
    message: str
    response: str

class ChatHistoryOut(ChatHistoryBase):
    id: int
    timestamp: datetime

    class Config:
        orm_mode = True




