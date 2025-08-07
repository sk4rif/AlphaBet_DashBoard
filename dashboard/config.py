from zoneinfo import ZoneInfo

# Dashboard Configuration
PAGE_TITLE = "AlphaBet Dashboard"
TIME_ZONE = ZoneInfo('Europe/London')
REFRESH_INTERVAL = 120  # seconds

# MongoDB Configuration
MONGO_URI = "mongodb+srv://CarlGrimaldi:AlphaBeta21$@alphabetcluster.x7lvc.mongodb.net/"
DB_NAME = "AlphaBet"
HISTORY_COLLECTION = "HISTORY"

# Plot Configuration
PLOT_COLUMNS = [
    'position_value', 'allocation',
    'probability', 'best_ask', 'adjustment', 'adjustment_ratio'
]

# Currency Configuration
CURRENCIES = ["BTC", "ETH", "SOL", "XRP"]
