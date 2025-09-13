# app/train_category_model.py
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Expanded labeled training data
data = [
    # Food
    ("Bought pizza", "Food"),
    ("Coffee at Starbucks", "Food"),
    ("Lunch at McDonald's", "Food"),
    ("Dinner at Italian restaurant", "Food"),
    ("Groceries from Walmart", "Food"),
    ("KFC chicken bucket", "Food"),
    ("Sushi order", "Food"),
    ("Bakery bread and cake", "Food"),
    ("Grocery shopping at Carrefour", "Food"),
    ("Burger King combo meal", "Food"),
    ("Food", "Food"),

    # Transport
    ("Uber ride", "Transport"),
    ("Taxi fare", "Transport"),
    ("Bus ticket", "Transport"),
    ("Train pass", "Transport"),
    ("Metro card recharge", "Transport"),
    ("Flight ticket", "Transport"),
    ("Gas refill for car", "Transport"),
    ("Bolt ride", "Transport"),
    ("Parking fee", "Transport"),
    ("Rental car payment", "Transport"),
    ("Transport", "Transport"),

    # Bills
    ("Paid electricity bill", "Bills"),
    ("Water bill", "Bills"),
    ("Gas bill payment", "Bills"),
    ("Monthly rent", "Bills"),
    ("Internet bill", "Bills"),
    ("Phone recharge", "Bills"),
    ("Heating bill", "Bills"),
    ("Cable TV subscription", "Bills"),
    ("Garbage collection fee", "Bills"),
    ("Insurance payment", "Bills"),
    ("Bills", "Bills"),

    # Entertainment
    ("Netflix subscription", "Entertainment"),
    ("Concert ticket", "Entertainment"),
    ("Cinema movie ticket", "Entertainment"),
    ("Spotify premium", "Entertainment"),
    ("Gaming console purchase", "Entertainment"),
    ("Theater play ticket", "Entertainment"),
    ("Sports event ticket", "Entertainment"),
    ("Amusement park entry", "Entertainment"),
    ("Video game purchase", "Entertainment"),
    ("YouTube premium", "Entertainment"),
    ("Entertainment", "Entertainment"),

    # Shopping
    ("New shoes from mall", "Shopping"),
    ("Amazon order", "Shopping"),
    ("Clothes shopping at Zara", "Shopping"),
    ("Bought iPhone from Apple Store", "Shopping"),
    ("Laptop purchase", "Shopping"),
    ("Gifts from souvenir shop", "Shopping"),
    ("Furniture from IKEA", "Shopping"),
    ("Handbag from Gucci", "Shopping"),
    ("Online shopping from eBay", "Shopping"),
    ("Cosmetics from Sephora", "Shopping"),
    ("Shopping", "Shopping"),

    # Other
    ("Donation to charity", "Other"),
    ("Doctor consultation fee", "Other"),
    ("School tuition fee", "Other"),
    ("Bought books", "Other"),
    ("Gym membership", "Other"),
    ("Pet food", "Other"),
    ("Hospital bill", "Other"),
    ("Stationery items", "Other"),
    ("Car repair service", "Other"),
    ("Baby diapers", "Other"),
    ("Other", "Other"),
]

texts, labels = zip(*data)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
pipeline.fit(texts, labels)

# Ensure models/ folder exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(pipeline, "models/category_model.pkl")

print("✅ Model trained and saved to models/category_model.pkl")
