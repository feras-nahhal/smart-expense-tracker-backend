import io
import csv
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from passlib.context import CryptContext
from . import models
from .nlp import detect_anomaly, predict_monthly_total
from .models import Goal
from .schemas import GoalCreate, ExpenseQuick
from .models import Category
from collections import defaultdict
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import numpy as np

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Default percentages per category
DEFAULT_CATEGORY_PERCENTAGES = {
    "Food": 0.4,
    "Shopping": 0.2,
    "Transport": 0.1,
    "Entertainment": 0.1,
    "Bills": 0.2,
    "Other": 0.1
}

# -----------------------------
# Password helpers
# -----------------------------
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# -----------------------------
# User CRUD
# -----------------------------
def create_user(db: Session, name: str, email: str, monthly_budget: float, password: str):
    hashed_password = get_password_hash(password)
    db_user = models.User(
        name=name,
        email=email,
        monthly_budget=monthly_budget,
        password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Automatically create budgets per category
    for category, pct in DEFAULT_CATEGORY_PERCENTAGES.items():
        cat_budget = models.Budget(
            user_id=db_user.id,
            category=category,
            monthly_limit=monthly_budget * pct
        )
        db.add(cat_budget)
    db.commit()

    return db_user

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.password):
        return None
    return user

# -----------------------------
# Expenses
# -----------------------------
MAX_EXPENSE_AMOUNT = 100_000  # ✅ prevent billion-dollar pizza

def create_expense(db: Session, user_id: int, amount: float, category: str, description: str, date=None, merchant=None):
    # ✅ Validation
    if amount <= 0 or amount > MAX_EXPENSE_AMOUNT:
        raise ValueError(f"❌ Invalid amount: {amount}. Must be between 0 and {MAX_EXPENSE_AMOUNT}.")

    category = (category or "Other").strip()
    cat_obj = get_category_by_name(db, category)
    if not cat_obj:
        category = "Other"

    if description and len(description) > 255:
        description = description[:255]
    if merchant and len(merchant) > 100:
        merchant = merchant[:100]

    exp = models.Expense(
        user_id=user_id,
        amount=amount,
        category=category,
        description=description,
        date=date or datetime.utcnow(),
        merchant=merchant
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


def list_expenses(db: Session, user_id: int):
    return db.query(models.Expense).filter(models.Expense.user_id == user_id).all()

# -----------------------------
# Chat
# -----------------------------
def create_chat(db: Session, user_id: int, message: str, response: str):
    chat = models.ChatHistory(user_id=user_id, message=message, response=response)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

# -----------------------------
# Summaries
# -----------------------------
def get_monthly_summary(db: Session, user_id: int):
    now = datetime.now()
    start = datetime(now.year, now.month, 1)
    end = datetime(now.year, now.month + 1, 1) if now.month < 12 else datetime(now.year + 1, 1, 1)

    expenses = db.query(models.Expense.category, func.sum(models.Expense.amount)) \
                 .filter(models.Expense.user_id == user_id, models.Expense.date >= start, models.Expense.date < end) \
                 .group_by(models.Expense.category).all()

    total = sum([amt for _, amt in expenses])
    summary = {"total": total, "categories": {cat: amt for cat, amt in expenses}}
    return summary

def get_total_expenses(db: Session, user_id, category, start_date, end_date):
    return db.query(func.sum(models.Expense.amount)).filter(
        models.Expense.user_id == user_id,
        models.Expense.category == category,
        models.Expense.date >= start_date,
        models.Expense.date <= end_date
    ).scalar() or 0.0

def get_monthly_totals(db: Session, user_id):
    start = datetime.today().replace(day=1)
    totals = db.query(models.Expense.category, func.sum(models.Expense.amount)).filter(
        models.Expense.user_id == user_id,
        models.Expense.date >= start
    ).group_by(models.Expense.category).all()
    return dict(totals)

def get_budget_limit(db: Session, user_id, category):
    budget = db.query(models.Budget).filter(
        models.Budget.user_id == user_id,
        models.Budget.category == category
    ).first()
    return budget.monthly_limit if budget else 0.0

def get_user_monthly_budget(db: Session, user_id):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    return user.monthly_budget if user else 0.0

# -----------------------------
# Monthly prediction
# -----------------------------
def predict_monthly_spending(db: Session, user_id):
    return predict_monthly_total(db, user_id)

# -----------------------------
# Budget recommendations
# -----------------------------
def get_user_budgets(db: Session, user_id: int) -> dict:
    """
    Return all budgets for a user as {category: monthly_limit}.
    """
    budgets = db.query(models.Budget).filter(models.Budget.user_id == user_id).all()
    return {b.category: b.monthly_limit for b in budgets}

def get_budget_recommendation(db: Session, user_id: int) -> str:
    """Generate budget optimization advice based on user spending."""
    budgets = get_user_budgets(db, user_id)  
    totals = get_monthly_totals(db, user_id) # {category: spent}

    if not budgets:
        return "⚠️ No budgets set. Please set category goals first."

    advice = []
    overspending = {}
    underspending = {}

    for cat, limit in budgets.items():
        spent = totals.get(cat, 0.0)

        if spent > limit:
            overspending[cat] = spent - limit
        elif spent < limit * 0.5:  # less than 50% spent → surplus
            underspending[cat] = limit - spent

    # Overspending advice
    for cat, over in overspending.items():
        advice.append(f"• You're ${over:.2f} over in {cat} — consider reducing.")

    # Reallocation advice
    for cat, surplus in underspending.items():
        advice.append(f"• {cat} has about ${surplus:.2f} unused — consider reallocating.")

    if not advice:
        return "✅ All categories are on track. Keep it up!"

    return "💡 Budget Advice:\n" + "\n".join(advice)

def optimize_budgets(db: Session, user_id: int, max_reduction_pct: float = 0.5) -> dict:
    """
    Simple smart optimizer (heuristic reallocation):
    - Finds overspent categories (spent > budget) and underspent categories (surplus).
    - Attempts to cover overspending by taking from available surplus up to max_reduction_pct of each category's budget.
    - Redistributes proportionally and returns suggested new budgets and transfer instructions.

    Args:
      db: SQLAlchemy Session
      user_id: target user
      max_reduction_pct: max fraction of a category's budget allowed to be reallocated (default 50%)

    Returns:
      dict containing:
        - original_budgets: {cat: limit}
        - totals: {cat: spent}
        - overspending: {cat: amount_over}
        - surplus: {cat: available_to_reallocate}
        - transfers: list of {from, to, amount}
        - suggested_budgets: {cat: new_limit}
        - summary: human-readable summary
    """
    budgets = get_user_budgets(db, user_id)
    if not budgets:
        return {"error": "No budgets found for user."}

    totals = get_monthly_totals(db, user_id)  # spent per category
    # compute overs and surplus capacities
    overs = {}
    surplus = {}
    for cat, limit in budgets.items():
        spent = totals.get(cat, 0.0)
        if spent > limit:
            overs[cat] = round(spent - limit, 2)
        else:
            # available pool we can potentially reallocate from this category:
            available = max(0.0, limit - spent)
            # enforce max_reduction_pct cap
            max_reducible = round(limit * float(max_reduction_pct), 2)
            available_to_reallocate = round(min(available, max_reducible), 2)
            if available_to_reallocate > 0:
                surplus[cat] = available_to_reallocate

    total_need = round(sum(overs.values()), 2)
    total_available = round(sum(surplus.values()), 2)

    suggested = budgets.copy()
    transfers = []

    if total_need == 0:
        return {
            "original_budgets": budgets,
            "totals": totals,
            "overspending": overs,
            "surplus": surplus,
            "transfers": [],
            "suggested_budgets": suggested,
            "summary": "No overspending detected — no optimization needed."
        }

    if total_available == 0:
        return {
            "original_budgets": budgets,
            "totals": totals,
            "overspending": overs,
            "surplus": surplus,
            "transfers": [],
            "suggested_budgets": suggested,
            "summary": f"Overspending total ${total_need:.2f} but no available surplus to reallocate."
        }

    # Approach:
    # - For each overspent category, compute share_of_need = overs[cat] / total_need
    # - Allocate available funds to each overspent category proportionally:
    #     allocated_to_cat = total_available * share_of_need
    #   (cap allocated_to_cat to overs[cat])
    # - Deduct actual contributions proportionally from surplus categories.
    remaining_available = total_available
    needed_left = total_need

    # Step A: compute target allocations for overspent categories (cap at overs)
    allocations = {}
    for cat, need in overs.items():
        share = need / total_need
        alloc = round(min(need, round(total_available * share, 2)), 2)
        allocations[cat] = alloc
        remaining_available -= alloc
        needed_left -= alloc

    # If due to rounding there's remaining_available but needed_left > 0,
    # allocate iteratively to categories still needing money.
    if needed_left > 0 and remaining_available > 0:
        for cat in overs:
            if needed_left <= 0 or remaining_available <= 0:
                break
            more_needed = round(overs[cat] - allocations.get(cat, 0.0), 2)
            if more_needed <= 0:
                continue
            give = round(min(more_needed, remaining_available), 2)
            allocations[cat] = round(allocations.get(cat, 0.0) + give, 2)
            remaining_available -= give
            needed_left -= give

    # Step B: subtract amounts proportionally from surplus categories
    # compute proportional factor of each surplus category vs total_available (original)
    orig_total_available = total_available
    surplus_contributions = {cat: 0.0 for cat in surplus}
    if orig_total_available > 0:
        for s_cat, avail in surplus.items():
            frac = avail / orig_total_available
            surplus_contributions[s_cat] = round(frac * (total_available - remaining_available), 2)

    # If due to rounding contributions don't sum exactly, adjust leftover from largest surplus
    contrib_sum = round(sum(surplus_contributions.values()), 2)
    target_sum = round(total_available - remaining_available, 2)
    diff = round(target_sum - contrib_sum, 2)
    if abs(diff) >= 0.01 and surplus_contributions:
        # apply diff to the largest surplus contributor
        largest = max(surplus_contributions.items(), key=lambda x: x[1])[0]
        surplus_contributions[largest] = round(surplus_contributions[largest] + diff, 2)

    # Build transfers from surplus -> overs based on proportional logic:
    # For each overspent category, we have allocations[cat]; we must allocate that
    # amount across surplus categories proportional to their contributions.
    for to_cat, alloc_amount in allocations.items():
        if alloc_amount <= 0:
            continue
        # allocate from each surplus category proportional to surplus_contributions
        contrib_total = round(sum(surplus_contributions.values()), 2)
        if contrib_total <= 0:
            break
        for from_cat, from_contrib in list(surplus_contributions.items()):
            if from_contrib <= 0:
                continue
            frac = from_contrib / contrib_total if contrib_total else 0
            take = round(min(from_contrib, round(alloc_amount * frac, 2)), 2)
            if take <= 0:
                continue
            transfers.append({"from": from_cat, "to": to_cat, "amount": take})
            # update bookkeeping
            surplus_contributions[from_cat] = round(surplus_contributions[from_cat] - take, 2)
            alloc_amount = round(alloc_amount - take, 2)
        # If still leftover (due to rounding), try to take from any remaining surplus
        if alloc_amount > 0:
            for from_cat, left in surplus_contributions.items():
                if left <= 0 or alloc_amount <= 0:
                    continue
                take = round(min(left, alloc_amount), 2)
                transfers.append({"from": from_cat, "to": to_cat, "amount": take})
                surplus_contributions[from_cat] = round(surplus_contributions[from_cat] - take, 2)
                alloc_amount = round(alloc_amount - take, 2)

    # Step C: compute suggested_budgets by applying transfers
    suggested = budgets.copy()
    for t in transfers:
        suggested[t["from"]] = round(suggested.get(t["from"], 0.0) - t["amount"], 2)
        suggested[t["to"]] = round(suggested.get(t["to"], 0.0) + t["amount"], 2)

    summary_lines = []
    summary_lines.append(f"Total overspending: ${total_need:.2f}")
    summary_lines.append(f"Available surplus: ${total_available:.2f}")
    if transfers:
        summary_lines.append("Proposed transfers:")
        for t in transfers:
            summary_lines.append(f"  • ${t['amount']:.2f} from {t['from']} → {t['to']}")
    else:
        summary_lines.append("No transfers possible (insufficient surplus).")

    return {
        "original_budgets": budgets,
        "totals": totals,
        "overspending": overs,
        "surplus": surplus,
        "transfers": transfers,
        "suggested_budgets": suggested,
        "summary": "\n".join(summary_lines)
    }


# -----------------------------
# Anomaly check
# -----------------------------
def check_expense_anomaly(db: Session, user_id, category, amount):
    if detect_anomaly(db, user_id, category, amount):
        return f"⚠️ This ${amount:.2f} expense in {category} is unusually high compared to your recent history!"
    return ""

# -----------------------------
# Default budgets
# -----------------------------
def create_default_budgets(db: Session, user_id: int, total_budget: float):
    from .models import Budget
    for cat, pct in DEFAULT_CATEGORY_PERCENTAGES.items():
        limit = total_budget * pct
        budget = Budget(user_id=user_id, category=cat, monthly_limit=limit)
        db.add(budget)
    db.commit()

# -----------------------------
# Goals
# -----------------------------
def create_or_update_goal(db: Session, goal: GoalCreate):
    existing = db.query(Goal).filter(
        Goal.user_id == goal.user_id,
        Goal.category == goal.category,
        Goal.period == goal.period
    ).first()

    if existing:
        existing.target_amount = goal.target_amount
        db.commit()
        db.refresh(existing)
        return existing

    new_goal = Goal(
        user_id=goal.user_id,
        category=goal.category.strip(),
        target_amount=goal.target_amount,
        period=goal.period,
    )
    db.add(new_goal)
    db.commit()
    db.refresh(new_goal)
    return new_goal


def get_goals(db: Session, user_id: int):
    return db.query(Goal).filter(Goal.user_id == user_id).all()

# -----------------------------
# Trends
# -----------------------------
def get_expense_trends(db: Session, user_id: int, period: str = "daily"):
    """
    Returns historical expenses per category for last 30 days (grouped by date).
    """
    start_date = datetime.utcnow() - timedelta(days=30)

    query = db.query(
        func.date(models.Expense.date).label("day"),
        models.Expense.category,
        func.sum(models.Expense.amount).label("total")
    ).filter(
        models.Expense.user_id == user_id,
        models.Expense.date >= start_date
    ).group_by(
        "day", models.Expense.category
    ).order_by("day").all()

    trends = {}
    for day, cat, amt in query:
        trends.setdefault(cat, []).append({"date": str(day), "amount": float(amt)})
    return trends

# -----------------------------
# Quick Expense
# -----------------------------
def add_expense_quick(db: Session, expense):
    """Quick add with same validation."""
    return create_expense(
        db,
        user_id=expense.user_id,
        amount=expense.amount,
        category=expense.category,
        description=expense.description,
        date=datetime.utcnow(),
        merchant=expense.merchant
    )

# -----------------------------
# Import expenses from CSV file (Improved & Flexible Dates)
# -----------------------------
from dateutil import parser as date_parser  

def import_csv(db: Session, user_id: int, file_bytes: bytes, alert_callback=None):
    """
    Import expenses from CSV file.
    Each row should be: date, description, amount
    - Uses dateutil to parse flexible date formats.
    - Skips invalid rows safely.
    - Returns summary of imported/skipped rows.
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    next(reader, None)  # skip header

    imported = 0
    skipped = 0

    for row in reader:
        if len(row) < 3:
            skipped += 1
            continue

        date_str, desc, amount_str = row[0].strip(), row[1].strip(), row[2].strip()

        # Parse amount
        try:
            amount = float(amount_str.replace("$", "").replace(",", ""))
        except ValueError:
            skipped += 1
            continue

        # Parse date with flexible parser
        try:
            dt = date_parser.parse(date_str, dayfirst=False, fuzzy=True)
        except Exception:
            dt = datetime.utcnow()

        # Predict category
        from . import nlp
        cat = getattr(nlp, "ml_classify_category", lambda t: None)(desc) or nlp.classify_category(desc)

        # Save expense
        create_expense(db, user_id, amount, cat, desc, dt)
        imported += 1

        # --- Proactive alert check ---
        monthly_total = get_total_expenses(db, user_id, cat, datetime.utcnow().replace(day=1), datetime.utcnow())
        budget_limit = get_budget_limit(db, user_id, cat) or 0.0
        if alert_callback and budget_limit > 0 and monthly_total > budget_limit:
            alert_callback(user_id, f"Over budget in {cat}: ${monthly_total:.2f}/${budget_limit:.2f}")

    return {
    "message": "CSV imported successfully",
    "imported": imported,
    "skipped": skipped
}

# -----------------------------
# create/get_category
# -----------------------------
def create_category(db: Session, name: str, keywords: str = ""):
    category = Category(name=name, keywords=keywords)
    db.add(category)
    db.commit()
    db.refresh(category)
    return category

def get_categories(db: Session):
    return db.query(Category).all()

def get_category_by_name(db: Session, name: str):
    return db.query(Category).filter(Category.name == name).first()

# -----------------------------
# create/get_chat_history
# -----------------------------

def save_chat_history(db: Session, user_id: int, message: str, response: str):
    chat = models.ChatHistory(
        user_id=user_id,
        message=message,
        response=response,
        timestamp=datetime.utcnow()
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

def get_chat_history(db: Session, user_id: int, limit: int = 50):
    return db.query(models.ChatHistory)\
             .filter(models.ChatHistory.user_id == user_id)\
             .order_by(models.ChatHistory.timestamp.desc())\
             .limit(limit).all()

# -----------------------------
# forecast_expenses
# -----------------------------

from pandas.errors import OutOfBoundsDatetime

def forecast_expenses(db: Session, user_id: int, months_ahead: int = 3):
    rows = db.query(models.Expense.date, models.Expense.amount).filter(
        models.Expense.user_id == user_id
    ).all()

    if not rows:
        return {"history": [], "forecast": []}

    # --- Step 1: group by month ---
    monthly = defaultdict(float)
    for dt, amt in rows:
        try:
            key = dt.strftime("%Y-%m")
            monthly[key] += amt
        except Exception:
            # skip weird dates
            continue

    # ✅ sanitize: only keep valid YYYY-MM
    monthly_series = [
        (k, v) for k, v in sorted(monthly.items(), key=lambda x: x[0])
        if k and len(k) == 7
    ]
    history = [{"date": k, "amount": v} for k, v in monthly_series]

    if not monthly_series:
        return {"history": history, "forecast": []}

    try:
        # ✅ explicitly set format so pandas doesn’t guess
        dates = pd.to_datetime(
            [k for k, _ in monthly_series],
            format="%Y-%m",
            errors="coerce"
        )
        values = [v for _, v in monthly_series]
        ts = pd.Series(values, index=dates).dropna()

        # ✅ Ensure sorted and set frequency
        ts = ts.sort_index()
        try:
            ts = ts.asfreq("M")
        except Exception:
            pass

    except OutOfBoundsDatetime:
        return {"history": history, "forecast": [], "error": "Invalid date range in data"}

    # --- Step 2: Forecast ---
    forecast_data = []
    try:
        if len(ts) >= 6:
            # Use ARIMA if enough data
            model = ARIMA(ts, order=(1, 1, 1))
            model_fit = model.fit()
            pred = model_fit.forecast(steps=months_ahead)

            for i, val in enumerate(pred):
                future_date = (ts.index[-1] + pd.DateOffset(months=i+1)).strftime("%Y-%m")
                forecast_data.append({"date": future_date, "predicted": float(val)})
        else:
            # Fallback to Linear Regression
            X = np.arange(len(ts)).reshape(-1, 1)
            y = ts.values
            model = LinearRegression().fit(X, y)
            future_X = np.arange(len(ts), len(ts) + months_ahead).reshape(-1, 1)
            preds = model.predict(future_X)

            for i, val in enumerate(preds):
                future_date = (ts.index[-1] + pd.DateOffset(months=i+1)).strftime("%Y-%m")
                forecast_data.append({"date": future_date, "predicted": float(val)})

    except Exception as e:
        return {"history": history, "forecast": [], "error": str(e)}

    return {"history": history, "forecast": forecast_data}
