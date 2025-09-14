# app/main.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from passlib.context import CryptContext
from . import models, schemas, database, crud, nlp, ocr
from .database import SessionLocal, engine
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# import the connection manager
from .websocket_manager import manager

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smart Expense Tracker API")

# -----------------------------
# CORS
# -----------------------------
origins = ["http://localhost:3000", "http://127.0.0.1:3000","https://smart-expense-tracker-frontend.vercel.app","https://smart-expense-tracker-frontend-8k9yweoqu.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Password hashing
# -----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# -----------------------------
# DB dependency
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# WebSocket helper wrapper
# -----------------------------
async def send_budget_alert(user_id: int, message: str):
    """
    Push proactive alert to a connected user. Uses the ConnectionManager.
    Keeps the same public API as before.
    """
    try:
        await manager.send_personal_message(user_id, f"🚨 {message}")
    except Exception:
        # swallow errors: caller may be background task
        pass

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Smart Expense Tracker API is running"}

# -----------------------------
# User registration & login
# -----------------------------
@app.post("/register")
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = crud.get_user_by_email(db, user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email exists")

    u = crud.create_user(db, user.name, user.email, user.monthly_budget, user.password)
    crud.create_default_budgets(db, u.id, user.monthly_budget)
    return {"user_id": u.id, "email": u.email}

@app.post("/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, user.email)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "user_id": db_user.id,
        "email": db_user.email,
        "monthly_budget": db_user.monthly_budget
    }

# -----------------------------
# Parse chat messages (REST)
# -----------------------------
@app.post("/chat/parse", response_model=schemas.ExpenseOut)
def parse_chat(payload: schemas.ExpenseIn, db: Session = Depends(get_db)):
    text = payload.text.lower()
    if "how much" in text or "spent this month" in text:
        response = crud.get_monthly_summary(db, payload.user_id)
        crud.save_chat_history(db, payload.user_id, payload.text, str(response))
        return response

    amt = nlp.extract_amount(text)
    if amt is None:
        raise HTTPException(status_code=400, detail="Amount not found")
    
    dt = nlp.extract_date(text)
    if not dt or dt.year < 2000 or dt.year > datetime.utcnow().year + 1:
        dt = datetime.utcnow()

    merch = nlp.extract_merchant(text)
    cat = (getattr(nlp, "ml_classify_category", lambda t: None)(text) or nlp.classify_category(text)).strip()
    exp = crud.create_expense(db, payload.user_id, amt, cat, text, dt, merch)

    crud.save_chat_history(db, payload.user_id, payload.text, f"Added {amt} to {cat}")
    return exp

# -----------------------------
# Get all expenses
# -----------------------------
@app.get("/expenses/{user_id}")
def get_expenses(user_id: int, db: Session = Depends(get_db)):
    return crud.list_expenses(db, user_id)

# -----------------------------
# WebSocket for real-time chat (multi-user isolated)
# -----------------------------
@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(ws: WebSocket, user_id: int):
    # Accept & register with manager
    await ws.accept()
    await manager.connect(user_id, ws)

    try:
        while True:
            data = await ws.receive_text()
            msg_lower = data.lower()
            amt = nlp.extract_amount(data)

            db: Session = SessionLocal()
            try:
                # ✅ Validate amount range before saving
                if amt is not None and (amt <= 0 or amt > 100_000):
                    msg = f"❌ Invalid amount: {amt}. Must be between 0 and 100,000."
                    await manager.send_personal_message(user_id, msg)
                    crud.save_chat_history(db, user_id, data, msg)
                    continue

                if amt is not None:
                    # --- Safe date fallback ---
                    dt = nlp.extract_date(data)
                    if not dt or dt.year < 2000 or dt.year > datetime.utcnow().year + 1:
                        dt = datetime.utcnow()

                    merch = nlp.extract_merchant(data)
                    cat = (
                        getattr(nlp, "ml_classify_category", lambda t: None)(data)
                        or nlp.classify_category(data)
                    ).strip()

                    exp = crud.create_expense(db, user_id, amt, cat, data, dt, merch)

                    db.expire_all()  # ensures fresh totals

                    today = datetime.utcnow()
                    week_start = today - timedelta(days=today.weekday())
                    month_start = today.replace(day=1)

                    weekly_total = crud.get_total_expenses(db, user_id, cat, week_start, today)
                    monthly_total = crud.get_total_expenses(db, user_id, cat, month_start, today)
                    budget_limit = crud.get_budget_limit(db, user_id, cat) or 0.0
                    percent = (monthly_total / budget_limit * 100) if budget_limit else 0

                    recommendation = ""
                    if budget_limit > 0 and monthly_total > budget_limit:
                        over = monthly_total - budget_limit
                        recommendation = f"⚠️ Over budget by ${over:.2f}! Consider reducing spending in {cat}."

                    anomaly_msg = crud.check_expense_anomaly(db, user_id, cat, amt)

                    response_msg = (
                        f"✅ Added ${amt:.2f} to {cat}.\n"
                        f"Weekly {cat} spending: ${weekly_total:.2f}/${budget_limit:.2f}\n"
                        f"Monthly {cat} spending: ${monthly_total:.2f}/${budget_limit:.2f} ({percent:.0f}%)\n"
                        f"{recommendation}"
                    )
                    if anomaly_msg:
                        response_msg += f"\n⚠️ {anomaly_msg}"

                    await manager.send_personal_message(user_id, response_msg)
                    crud.save_chat_history(db, user_id, data, response_msg)

                    # Goal alerts
                    goals = crud.get_goals(db, user_id)
                    for g in goals:
                        monthly_total_goal = crud.get_total_expenses(
                            db, user_id, g.category, datetime.utcnow().replace(day=1), datetime.utcnow()
                        )
                        if monthly_total_goal >= g.target_amount:
                            goal_msg = (
                                f"⚠️ Goal reached for {g.category}: "
                                f"${monthly_total_goal:.2f} / ${g.target_amount:.2f}"
                            )
                            await manager.send_personal_message(user_id, goal_msg)
                            crud.save_chat_history(db, user_id, data, goal_msg)

                    continue

                # Monthly summary
                elif "how much" in msg_lower or "spend" in msg_lower:
                    totals = crud.get_monthly_totals(db, user_id)
                    response_lines = [
                        f"📊 {datetime.utcnow():%B} total spending: ${sum(totals.values()):.2f}"
                    ]
                    for cat, amt in totals.items():
                        budget_limit = crud.get_budget_limit(db, user_id, cat)
                        percent = (amt / budget_limit * 100) if budget_limit else 0
                        response_lines.append(f"• {cat}: ${amt:.2f} ({percent:.0f}%)")
                    all_on_track = all(
                        (amt <= (crud.get_budget_limit(db, user_id, cat) or float('inf')))
                        for cat, amt in totals.items()
                    )
                    response_lines.append(
                        "💡 You're on track with all budgets!"
                        if all_on_track else "⚠️ Some categories exceeded budget."
                    )
                    final_msg = "\n".join(response_lines)
                    await manager.send_personal_message(user_id, final_msg)
                    crud.save_chat_history(db, user_id, data, final_msg)
                    continue

                # Predictions
                elif "go over budget" in msg_lower:
                    predicted_total = crud.predict_monthly_spending(db, user_id)
                    user_budget = crud.get_user_monthly_budget(db, user_id)
                    if predicted_total > user_budget:
                        over = predicted_total - user_budget
                        pred_msg = (
                            f"📈 Prediction: You'll spend ~${predicted_total:.2f}\n"
                            f"⚠️ ${over:.2f} over your ${user_budget:.2f} budget\n"
                            f"💡 Reduce expenses in high categories."
                        )
                        await manager.send_personal_message(user_id, pred_msg)
                        crud.save_chat_history(db, user_id, data, pred_msg)
                    else:
                        pred_msg = f"📈 Prediction: On track! Likely total: ${predicted_total:.2f}"
                        await manager.send_personal_message(user_id, pred_msg)
                        crud.save_chat_history(db, user_id, data, pred_msg)
                    continue

                # Budget advice
                elif "budget" in msg_lower or "save" in msg_lower:
                    rec = crud.get_budget_recommendation(db, user_id)
                    await manager.send_personal_message(user_id, f"💡 Budget Advice:\n{rec}")
                    crud.save_chat_history(db, user_id, data, f"Budget Advice: {rec}")
                    continue

                # Fallback
                else:
                    fallback_msg = (
                        "❌ Couldn't detect an amount or query.\n"
                        "Try: 'I spent $5 on coffee.', 'How much did I spend this month?', or ask for 'budget' advice."
                    )
                    await manager.send_personal_message(user_id, fallback_msg)
                    crud.save_chat_history(db, user_id, data, fallback_msg)
            finally:
                db.close()

    except WebSocketDisconnect:
        await manager.disconnect(user_id)
        print(f"User {user_id} disconnected")

# -----------------------------
# Upload receipt (OCR)
# -----------------------------
@app.post("/upload/receipt")
async def upload_receipt(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    contents = await file.read()

    text = ocr.extract_text_from_bytes(contents)
    amount = nlp.extract_amount_from_receipt(text) or nlp.extract_amount(text)
    if not amount:
        raise HTTPException(status_code=400, detail="Could not extract amount from receipt")

    dt = nlp.extract_date(text) or datetime.utcnow()
    merch = nlp.extract_merchant(text)
    cat = getattr(nlp, "ml_classify_category", lambda t: None)(text) or nlp.classify_category(text)

    expense = crud.create_expense(db, user_id, amount, cat, description=text, date=dt, merchant=merch)

    monthly_total = crud.get_total_expenses(db, user_id, cat, datetime.utcnow().replace(day=1), datetime.utcnow())
    budget_limit = crud.get_budget_limit(db, user_id, cat) or 0.0
    if budget_limit > 0 and monthly_total > budget_limit:
        background_tasks.add_task(send_budget_alert, user_id, f"Over budget in {cat}: ${monthly_total:.2f}/${budget_limit:.2f}")

    goals = crud.get_goals(db, user_id)
    for g in goals:
        monthly_total = crud.get_total_expenses(db, user_id, g.category, datetime.utcnow().replace(day=1), datetime.utcnow())
        if monthly_total >= g.target_amount:
            background_tasks.add_task(send_budget_alert, user_id, f"Goal reached for {g.category}: ${monthly_total:.2f}/${g.target_amount:.2f}")

    return {"id": expense.id, "amount": amount, "category": cat, "merchant": merch, "raw_text": text}

# -----------------------------
# Update budget per category
# -----------------------------
@app.post("/budget/update/{user_id}")
def update_budget(user_id: int, category: str = Body(...), monthly_limit: float = Body(...), db: Session = Depends(get_db)):
    budget = db.query(models.Budget).filter(models.Budget.user_id==user_id, models.Budget.category==category).first()
    if not budget:
        budget = models.Budget(user_id=user_id, category=category, monthly_limit=monthly_limit)
        db.add(budget)
    else:
        budget.monthly_limit = monthly_limit
    db.commit()
    return {"category": category, "monthly_limit": monthly_limit}

# -----------------------------
# Goals endpoints
# -----------------------------
@app.post("/goals")
def create_goal(goal: schemas.GoalCreate, db: Session = Depends(get_db)):
    return crud.create_or_update_goal(db, goal)

@app.get("/goals/{user_id}")
def list_goals(user_id: int, db: Session = Depends(get_db)):
    return crud.get_goals(db, user_id)

@app.get("/goals/alerts/{user_id}")
def goal_alerts(user_id: int, db: Session = Depends(get_db)):
    goals = crud.get_goals(db, user_id)
    alerts = []
    for g in goals:
        monthly_total = crud.get_total_expenses(
            db, user_id, g.category, datetime.utcnow().replace(day=1), datetime.utcnow()
        )
        if monthly_total >= g.target_amount:
            alerts.append(f"⚠️ Goal reached for {g.category}: ${monthly_total:.2f} / ${g.target_amount:.2f}")
    return alerts

# -----------------------------
# Quick expense entry (non-chat)
# -----------------------------
from fastapi import HTTPException

@app.post("/expense/quick")
def quick_expense(
    expense: schemas.ExpenseQuick,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    # ✅ Input validation for amount
    if expense.amount <= 0 or expense.amount > 100_000:
        msg = f"❌ Invalid amount: {expense.amount}. Must be between 0 and 100,000."
        raise HTTPException(status_code=400, detail=msg)

    # Save expense
    exp = crud.add_expense_quick(db, expense)

    # Budget overspending check
    monthly_total = crud.get_total_expenses(
        db, expense.user_id, exp.category, datetime.utcnow().replace(day=1), datetime.utcnow()
    )
    budget_limit = crud.get_budget_limit(db, expense.user_id, exp.category) or 0.0
    if budget_limit > 0 and monthly_total > budget_limit:
        if background_tasks:
            background_tasks.add_task(
                send_budget_alert,
                expense.user_id,
                f"Over budget in {exp.category}: ${monthly_total:.2f}/${budget_limit:.2f}"
            )

    # Goal checks
    goals = crud.get_goals(db, expense.user_id)
    alerts = []
    for g in goals:
        monthly_total = crud.get_total_expenses(
            db, expense.user_id, g.category, datetime.utcnow().replace(day=1), datetime.utcnow()
        )
        if monthly_total >= g.target_amount:
            msg = f"⚠️ Goal reached for {g.category}: ${monthly_total:.2f} / ${g.target_amount:.2f}"
            alerts.append(msg)
            if background_tasks:
                background_tasks.add_task(send_budget_alert, expense.user_id, msg)

    return {"expense": exp, "alerts": alerts}

# -----------------------------
# Budget vs Actual
# -----------------------------
@app.get("/budget/actuals/{user_id}")
def budget_vs_actual(user_id: int, db: Session = Depends(get_db)):
    budgets = db.query(models.Budget).filter(models.Budget.user_id == user_id).all()
    actuals = crud.get_monthly_totals(db, user_id)

    return [
        {
            "category": b.category,
            "budget": b.monthly_limit,
            "actual": actuals.get(b.category, 0.0)
        }
        for b in budgets
    ]

# -----------------------------
# budget_optimize
# -----------------------------
@app.get("/budget/optimize/{user_id}")
def budget_optimize(user_id: int, db: Session = Depends(get_db), max_reduction_pct: float = 0.5):
    result = crud.optimize_budgets(db, user_id, max_reduction_pct=max_reduction_pct)
    return result

# -----------------------------
# Spending Insights
# -----------------------------
@app.get("/insights/{user_id}")
def spending_insights(user_id: int, db: Session = Depends(get_db)):
    totals = crud.get_monthly_totals(db, user_id)
    if not totals:
        return {"message": "No spending data yet."}

    insights = []
    biggest = max(totals, key=totals.get)
    insights.append(f"📊 Biggest spending category: {biggest} (${totals[biggest]:.2f})")

    avg = sum(totals.values()) / len(totals)
    for cat, amt in totals.items():
        if amt > avg * 1.5:
            insights.append(f"⚠️ {cat} spending (${amt:.2f}) is much higher than average (${avg:.2f})")

    predicted = crud.predict_monthly_spending(db, user_id)
    user_budget = crud.get_user_monthly_budget(db, user_id)
    if predicted > user_budget:
        insights.append(f"🚨 On track to overspend by ${(predicted - user_budget):.2f}")

    return {"insights": insights}

# -----------------------------
# Expense trends / dashboard
# -----------------------------
@app.get("/expenses/trends/{user_id}")
def expense_trends(user_id: int, db: Session = Depends(get_db)):
    return crud.get_expense_trends(db, user_id)

# -----------------------------
# Bulk CSV import
# -----------------------------
@app.post("/upload/csv")
async def upload_csv(
    user_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    contents = await file.read()

    def schedule_alert(uid, msg):
        background_tasks.add_task(send_budget_alert, uid, msg)

    result = crud.import_csv(db, user_id, contents, alert_callback=schedule_alert)
    return result

# -----------------------------
# Chat history endpoint
# -----------------------------
@app.get("/chat/history/{user_id}", response_model=list[schemas.ChatHistoryOut])
def chat_history(user_id: int, db: Session = Depends(get_db)):
    return crud.get_chat_history(db, user_id)

# -----------------------------
# Scheduler startup (AsyncIOScheduler)
# -----------------------------
@app.on_event("startup")
async def start_scheduler():
    scheduler = AsyncIOScheduler()

    @scheduler.scheduled_job("interval", minutes=1)
    async def job():
        db = SessionLocal()
        try:
            users = db.query(models.User).all()
            for u in users:
                # --- Budget checks ---
                totals = crud.get_monthly_totals(db, u.id)
                for cat, spent in totals.items():
                    limit = crud.get_budget_limit(db, u.id, cat)
                    if limit > 0 and spent > limit:
                        await send_budget_alert(u.id, f"⚠️ Over budget in {cat}: ${spent:.2f}/${limit:.2f}")

                # --- Goal checks ---
                goals = crud.get_goals(db, u.id)
                for g in goals:
                    monthly_total = crud.get_total_expenses(
                        db, u.id, g.category, datetime.utcnow().replace(day=1), datetime.utcnow()
                    )
                    if monthly_total >= g.target_amount:
                        await send_budget_alert(
                            u.id,
                            f"🎯 Goal reached for {g.category}: ${monthly_total:.2f}/${g.target_amount:.2f}"
                        )
        finally:
            db.close()

    scheduler.start()

# -----------------------------
# Seed Default Categories
# -----------------------------
@app.post("/categories", response_model=schemas.CategoryOut)
def create_category(category: schemas.CategoryCreate, db: Session = Depends(get_db)):
    existing = crud.get_category_by_name(db, category.name)
    if existing:
        raise HTTPException(status_code=400, detail="Category already exists")
    return crud.create_category(db, category.name, category.keywords or "")

@app.get("/categories", response_model=list[schemas.CategoryOut])
def list_categories(db: Session = Depends(get_db)):
    return crud.get_categories(db)


@app.on_event("startup")
def seed_categories():
    db = SessionLocal()
    defaults = [
        ("Food", "food,restaurant,dinner,coffee"),
        ("Shopping", "shop,clothes,amazon,store"),
        ("Transport", "bus,train,taxi,uber"),
        ("Bills", "electricity,water,gas,internet"),
        ("Entertainment", "movie,netflix,game,concert"),
        ("Other", ""),
    ]
    for name, keywords in defaults:
        if not crud.get_category_by_name(db, name):
            crud.create_category(db, name, keywords)
    db.close()

# -----------------------------
# Global Error Handlers
# -----------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc)}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "details": exc.errors()}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

# -----------------------------
# Apply Suggested Budgets
# -----------------------------
@app.post("/budget/optimize/apply/{user_id}")
def apply_optimized_budgets(user_id: int, db: Session = Depends(get_db)):
    from .crud import optimize_budgets
    result = optimize_budgets(db, user_id)

    suggested = result.get("suggested_budgets", {})
    if not suggested:
        raise HTTPException(status_code=400, detail="No suggested budgets found")

    for cat, new_limit in suggested.items():
        budget = db.query(models.Budget).filter(
            models.Budget.user_id == user_id,
            models.Budget.category == cat
        ).first()
        if budget:
            budget.monthly_limit = new_limit
        else:
            budget = models.Budget(
                user_id=user_id,
                category=cat,
                monthly_limit=new_limit
            )
            db.add(budget)

    db.commit()
    return {"message": "✅ Suggested budgets applied", "new_budgets": suggested}

# -----------------------------
# Forecast endpoint
# -----------------------------
@app.get("/forecast/{user_id}")
def forecast(user_id: int, months: int = 3, db: Session = Depends(get_db)):
    return crud.forecast_expenses(db, user_id, months_ahead=months)
