# ml_service/normalization/category_classifier.py

from sqlalchemy.orm import Session
from app.models import ShelfLifeReference

CATEGORY_KEYWORDS = {
    "Produce": [
        "berries", "berry", "apple", "apples", "orange", "mango", "banana",
        "lettuce", "tomato", "onion", "garlic", "potato", "carrot", "spinach",
        "cucumber", "pepper", "mushroom", "zucchini", "broccoli", "cauliflower",
        "cabbage", "eggplant", "okra", "peas", "beans", "corn", "celery",
        "coriander", "mint", "lemon", "lime", "grapes", "guava", "papaya",
        "kiwi", "pear", "plum", "apricot", "cherry", "pomegranate", "watermelon",
        "cantaloupe", "pineapple", "passionfruit", "dragonfruit", "starfruit",
        "persimmon", "kale", "asparagus",
    ],
    "Dairy": [
        "milk", "cheese", "yogurt", "dahi", "butter", "cream", "eggs",
        "paneer", "lassi", "ghee", "curd", "brie", "feta", "ricotta",
        "evaporated", "kefir", "buttermilk", "camembert",
    ],
    "Meat": [
        "chicken", "beef", "mutton", "lamb", "pork", "salmon", "tuna",
        "shrimp", "bacon", "sausage", "turkey", "qeema", "keema",
        "gosht", "murg", "seekh", "kebab", "duck", "venison", "goat",
        "crab", "lobster", "clam", "oyster",
    ],
    "Pantry": [
        "pasta", "rice", "flour", "sugar", "salt", "oil", "sauce", "ketchup",
        "mayo", "mustard", "bread", "oats", "lentils", "chickpeas", "beans",
        "atta", "maida", "dal", "masala", "spice", "vinegar", "honey",
        "peanut", "jam", "basmati", "chawal", "couscous", "millet", "barley",
        "rye", "brown sugar", "maple", "tahini", "molasses", "coconut oil",
    ],
    "Frozen": [
        "frozen", "frzn", "ice cream", "popsicle", "nuggets", "dumplings",
        "fish fillets", "broccoli", "mango", "strawberries",
    ],
    "Beverages": [
        "juice", "milk", "soda", "coffee", "tea", "water", "chai", "lassi",
        "pineapple", "cranberry", "energy drink", "herbal", "green tea",
        "instant coffee",
    ],
    "Bakery": [
        "bread", "naan", "roti", "paratha", "chapati", "croissant", "bagel",
        "muffin", "cake", "roll", "donut", "scone", "brownie", "cupcake",
        "tart", "eclair",
    ],
    "Snacks": [
        "chips", "pretzel", "trail mix", "popcorn", "granola", "rice cracker",
        "cheese puff", "jerky", "fruit leather", "seaweed", "cracker",
    ],
    "Condiments & Sauces": [
        "hummus", "tartar", "ranch", "worcestershire", "salsa", "harissa",
        "gochujang", "bbq", "pesto", "fish sauce", "chili sauce",
    ],
    "Prepared Meals": [
        "sandwich", "pizza slice", "sushi", "shawarma", "spring roll",
        "pasta salad", "chicken curry", "lasagna", "biryani",
    ],
    "Breakfast Foods": [
        "cornflakes", "muesli", "pancake", "waffle", "bagel with cream cheese",
        "breakfast burrito", "scrambled eggs",
    ],
    "Confectionery": [
        "chocolate", "candy", "marshmallow", "fudge", "truffle", "gummy",
        "halva",
    ],
    "Baby Food": [
        "formula", "baby cereal", "puree", "pouch", "toddler snack", "rusks",
    ],
}


def assign_category(canonical_name: str, db: Session) -> str:
    """
    Assign category to a canonical food name.

    Priority:
    1. Exact match in shelf_life_reference (most reliable)
    2. Keyword classifier fallback (approximate)
    3. "Other" if no match

    Returns category string.
    """
    # Pass 1: DB lookup
    ref = db.query(ShelfLifeReference).filter_by(canonical_name=canonical_name).first()
    if ref:
        return ref.category

    # Pass 2: keyword classifier
    name_lower = canonical_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category

    return "Other"