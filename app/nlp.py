# app/nlp.py
import re
from datetime import datetime, timedelta
from dateutil import parser   # flexible natural language parsing
from sklearn.ensemble import IsolationForest
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
import spacy
import joblib

from . import crud

# ==============================
# Load NLP + ML models
# ==============================
# spaCy for NER
try:
    nlp_spacy = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded: en_core_web_sm")
except OSError:
    nlp_spacy = None
    print("⚠️ spaCy model not installed, some NLP features may be weaker")

# scikit-learn model for category classification (optional)
try:
    category_model = joblib.load("models/category_model.pkl")
    print("✅ ML Category Classifier loaded (models/category_model.pkl)")
except Exception:
    category_model = None
    print("⚠️ Using keyword fallback, ML model not loaded")


# --- Amount extraction ---
def extract_amount(text: str):
    """Extract the first amount (via spaCy or regex). Returns float or None."""
    if not text:
        return None

    # Try spaCy MONEY entities first (if available)
    if nlp_spacy:
        try:
            doc = nlp_spacy(text)
            for ent in doc.ents:
                if ent.label_ == "MONEY":
                    try:
                        return float(ent.text.replace("$", "").replace(",", "").strip())
                    except ValueError:
                        pass
        except Exception:
            pass

    # fallback regex: capture numbers possibly preceded by $
    match = re.search(r"\$?\s*([0-9]{1,3}(?:[,0-9]*)(?:\.\d+)?|\d+(?:\.\d+)?)", text.replace(",", ""))
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def extract_amount_from_receipt(text: str):
    text_lower = (text or "").lower()
    lines = text_lower.split("\n")
    # Look for typical receipt totals
    keywords = ['subtotal', 'total', 'amount due', 'balance', 'grand total', 'amount']
    for line in reversed(lines):  # check bottom-up (totals usually at bottom)
        line_stripped = line.strip()
        for keyword in keywords:
            if keyword in line_stripped:
                numbers = re.findall(r"\$?([\d,.]+)", line_stripped)
                if numbers:
                    try:
                        amount = float(numbers[-1].replace(",", ""))
                        return amount
                    except ValueError:
                        continue
    # fallback: last numeric token in whole text
    numbers = re.findall(r"\$?([\d,.]+)", text_lower)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


# --- Date extraction ---
def extract_date(text: str):
    """
    Extract a date from input.
    - Supports relative terms ('yesterday', 'last week') and explicit formats.
    - Avoids interpreting pure numeric amounts as years/dates (fixes dead-zone bug).
    - If resulting date looks unrealistic (year < 2000 or > next year), return None.
    """
    if not text:
        return None

    txt = text.lower().strip()
    today = datetime.utcnow()

    # quick relative terms
    if "yesterday" in txt:
        return today - timedelta(days=1)
    if "today" in txt:
        return today
    if "last week" in txt:
        return today - timedelta(days=7)
    if "last month" in txt:
        return today - timedelta(days=30)

    # Avoid parsing if text contains only an amount or only digits (these are likely amounts)
    # e.g. "Food 1000" or "1000" would otherwise parse 1000 as a year.
    # If there's a clear money token, skip trying to parse a date.
    if re.fullmatch(r"\$?\s*\d+(\.\d+)?", txt) or re.search(r"\$\s*\d", txt):
        return None

    # If there are tokens that look like dates (month names, slashes, hyphens), attempt parsing
    likely_date_token = re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|/|-)\b", txt, re.I)
    try:
        if nlp_spacy:
            # Use spaCy date entities if possible
            doc = nlp_spacy(text)
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    try:
                        parsed = parser.parse(ent.text, fuzzy=True, default=today)
                        # guard: parsed years like 1000 are suspicious
                        if parsed.year < 2000 or parsed.year > today.year + 1:
                            return None
                        return parsed
                    except Exception:
                        continue
        # If there's a likely date token or alphanumeric date-like text, try parser
        if likely_date_token or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", txt):
            parsed = parser.parse(text, fuzzy=True, default=today)
            if parsed.year < 2000 or parsed.year > today.year + 1:
                return None
            return parsed
    except Exception:
        pass

    # Fall back: no valid date detected
    return None


# --- Merchant extraction ---
def extract_merchant(text: str):
    """Try to get merchant (via spaCy OR fallback regex 'at <merchant>')."""
    if not text:
        return None

    if nlp_spacy:
        try:
            doc = nlp_spacy(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    return ent.text.strip()
        except Exception:
            pass

    match = re.search(r"at ([\w\s&\.-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


# --- Expense categorization ---
def classify_category(text: str):
    """Classify expense category (ML if available, else keyword rules)."""
    if not text:
        return "Other"

    if category_model:
        try:
            return category_model.predict([text])[0]
        except Exception:
            pass

    # fallback rules
    t = text.lower()
    if any(word in t for word in ["food", "lunch", "coffee", "restaurant", "grocer", "meal"]):
        return "Food"
    if any(word in t for word in ["bus", "taxi", "uber", "transport", "metro", "train"]):
        return "Transport"
    if any(word in t for word in ["walmart", "shopping", "clothes", "mall", "store", "amazon"]):
        return "Shopping"
    if any(word in t for word in ["bill", "electricity", "water", "internet", "rent"]):
        return "Bills"
    if any(word in t for word in ["movie", "cinema", "game", "entertainment", "netflix"]):
        return "Entertainment"
    return "Other"


# --- Weekly/monthly helpers ---
def weekly_category_spending(db: Session, user_id: int, category: str):
    start_date = datetime.utcnow() - timedelta(days=7)
    total = (
        db.query(func.sum(crud.models.Expense.amount))
        .filter(crud.models.Expense.user_id == user_id,
                crud.models.Expense.category == category,
                crud.models.Expense.date >= start_date)
        .scalar()
    )
    return total or 0.0


def monthly_category_spending(db: Session, user_id: int, category: str):
    start_date = datetime.utcnow().replace(day=1)
    total = (
        db.query(func.sum(crud.models.Expense.amount))
        .filter(crud.models.Expense.user_id == user_id,
                crud.models.Expense.category == category,
                crud.models.Expense.date >= start_date)
        .scalar()
    )
    return total or 0.0


def monthly_total_spending(db: Session, user_id: int):
    start_date = datetime.utcnow().replace(day=1)
    total = (
        db.query(func.sum(crud.models.Expense.amount))
        .filter(crud.models.Expense.user_id == user_id,
                crud.models.Expense.date >= start_date)
        .scalar()
    )
    return total or 0.0


# --- Monthly spending prediction ---
def predict_monthly_total(db: Session, user_id: int):
    import pandas as pd
    from datetime import datetime
    from . import models

    # --- Fetch expense history ---
    expenses = db.query(models.Expense).filter(
        models.Expense.user_id == user_id
    ).all()

    if not expenses:
        return 0.0

    # --- Build DataFrame for forecasting ---
    df = pd.DataFrame([{"ds": e.date, "y": e.amount} for e in expenses])

    # Aggregate daily totals
    df = df.groupby("ds").sum().reset_index()

    try:
        # --- Try Prophet model ---
        from prophet import Prophet
        model = Prophet()
        model.fit(df)

        # Forecast next 30 days
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Sum predicted values for the next 30 days
        next_month_forecast = forecast.tail(30)["yhat"].sum()
        return float(next_month_forecast)

    except Exception:
        # Fallback: linear extrapolation based on days so far this month
        now = datetime.utcnow()
        start_month = now.replace(day=1)
        total_so_far = monthly_total_spending(db, user_id)
        days_passed = (now - start_month).days + 1

        if now.month == 12:
            next_month = start_month.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = start_month.replace(month=now.month + 1, day=1)

        days_in_month = (next_month - start_month).days
        predicted = total_so_far / max(1, days_passed) * days_in_month
        return float(predicted)


# --- Anomaly detection ---

def detect_anomaly(db, user_id, category, amount):
    from . import models
    import numpy as np
    from sklearn.ensemble import IsolationForest

    expenses = db.query(models.Expense.amount)\
                 .filter(models.Expense.user_id == user_id,
                         models.Expense.category == category).all()

    amounts = [float(x[0]) for x in expenses if x[0] is not None]

    if len(amounts) < 5:
        return False

    arr = np.array(amounts)

    # --- Z-Score ---
    mean, std = np.mean(arr), np.std(arr)
    if std > 0 and abs((amount - mean) / std) > 3:
        return True

    # --- IQR ---
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    if amount < lower or amount > upper:
        return True

    # --- Isolation Forest ---
    try:
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(arr.reshape(-1, 1))
        if model.predict([[amount]])[0] == -1:
            return True
    except Exception as e:
        print("⚠️ IsolationForest failed:", e)

    # --- Relative Jump Detector (NEW) ---
    last_five = arr[-5:] if len(arr) >= 5 else arr
    avg_recent = np.mean(last_five)
    if avg_recent > 0 and amount > avg_recent * 3:  # >3x recent average
        return True

    return False
