"""
API Trigger Scheduler - Simulasi transaksi masuk secara real-time
Mengirim prediction requests ke inference API setiap interval waktu
dengan data simulasi yang realitik
"""

import json
import random
import time
from datetime import datetime
import requests
import schedule
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API endpoint
API_URL = "http://localhost:5000/predict"

# Simulasi dataset untuk realistic transactions
MERCHANTS = [
    ("Amazon", 2),      # online retail
    ("Starbucks", 3),   # food & drink
    ("Gas Station", 4), # fuel
    ("Hospital", 8),    # healthcare
    ("Restaurant", 5),  # dining
    ("Grocery", 6),     # supermarket
    ("ATM", 7),         # cash withdrawal
    ("Airplane", 9),    # travel
    ("Hotel", 10),      # accommodation
    ("Gaming", 11),     # entertainment
]

TRANSACTION_HOURS = list(range(24))
AMOUNTS = {
    "small": (10, 100),       # daily purchases
    "medium": (100, 500),     # regular shopping
    "large": (500, 2000),     # big purchases
    "huge": (2000, 10000),    # unusual/fraud
}

def generate_realistic_transaction():
    """Generate realistic transaction data for prediction."""
    
    # Determine transaction type
    rand = random.random()
    
    # 95% normal transactions, 5% suspicious
    if rand < 0.95:
        # Normal transaction
        amount_type = random.choices(
            ["small", "medium", "large"],
            weights=[60, 30, 10]
        )[0]
        foreign = random.choices([0, 1], weights=[85, 15])[0]
        location_mismatch = 0 if foreign == 0 else random.choices([0, 1], weights=[70, 30])[0]
        device_trust = random.uniform(0.75, 1.0)
        velocity = random.randint(0, 5)
    else:
        # Suspicious transaction (higher fraud probability)
        amount_type = random.choices(
            ["medium", "large", "huge"],
            weights=[20, 30, 50]
        )[0]
        foreign = 1
        location_mismatch = random.choices([0, 1], weights=[30, 70])[0]
        device_trust = random.uniform(0.0, 0.4)
        velocity = random.randint(5, 15)
    
    # Generate amount
    min_amt, max_amt = AMOUNTS[amount_type]
    amount = round(random.uniform(min_amt, max_amt), 2)
    
    # Transaction time
    hour = random.choice(TRANSACTION_HOURS)
    
    # Cardholder age
    age = random.randint(18, 80)
    age_group = 0 if age < 25 else (1 if age < 40 else (2 if age < 60 else 3))
    
    # Merchant
    merchant, merchant_code = random.choice(MERCHANTS)
    
    # Amount bin
    amount_bin = 0 if amount < 50 else (1 if amount < 200 else (2 if amount < 500 else (3 if amount < 1000 else (4 if amount < 2000 else 5))))
    
    # Time period
    if 6 <= hour < 12:
        time_period = 0  # morning
    elif 12 <= hour < 18:
        time_period = 1  # afternoon
    else:
        time_period = 2  # night
    
    return {
        "amount": amount,
        "transaction_hour": hour,
        "foreign_transaction": foreign,
        "location_mismatch": location_mismatch,
        "device_trust_score": round(device_trust, 2),
        "velocity_last_24h": velocity,
        "cardholder_age": age,
        "merchant_category_encoded": merchant_code,
        "amount_bin_encoded": amount_bin,
        "age_group_encoded": age_group,
        "time_period_encoded": time_period,
        "_merchant_name": merchant,  # for logging only
    }


def trigger_api(transaction_data):
    """Send transaction to inference API."""
    try:
        response = requests.post(
            API_URL,
            json=transaction_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Log transaction
            is_fraud = "🚨 FRAUD" if result['fraud'] else "✓ OK"
            log_msg = (
                f"{is_fraud} | Amount: ${transaction_data['amount']:.2f} | "
                f"Merchant: {transaction_data.get('_merchant_name', 'Unknown')} | "
                f"Confidence: {result['confidence']:.4f} | "
                f"Hour: {transaction_data['transaction_hour']:02d}:00"
            )
            logger.info(log_msg)
            
            return True
        else:
            logger.error(f"API Error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to API at http://localhost:5000")
        return False
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False


def job_trigger():
    """Job function to be scheduled."""
    transaction = generate_realistic_transaction()
    trigger_api(transaction)


def schedule_transactions(interval_seconds=2, duration_minutes=None):
    """
    Schedule API triggers at regular intervals.
    
    Args:
        interval_seconds: Time between predictions (default 2 seconds)
        duration_minutes: Run for X minutes (None = infinite)
    """
    
    logger.info("=" * 70)
    logger.info("FRAUD DETECTION API TRIGGER SCHEDULER")
    logger.info("=" * 70)
    logger.info(f"API Endpoint: {API_URL}")
    logger.info(f"Interval: {interval_seconds} seconds between predictions")
    if duration_minutes:
        logger.info(f"Duration: {duration_minutes} minutes")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 70)
    
    # Schedule job
    schedule.every(interval_seconds).seconds.do(job_trigger)
    
    start_time = time.time()
    predictions_sent = 0
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(0.1)
            
            # Check duration limit
            if duration_minutes:
                elapsed = (time.time() - start_time) / 60
                if elapsed >= duration_minutes:
                    logger.info(f"\nDuration limit reached ({duration_minutes} minutes)")
                    break
            
            predictions_sent = sum(1 for job in schedule.jobs if job.last_run)
            
    except KeyboardInterrupt:
        logger.info("\n\nScheduler stopped by user")
    finally:
        logger.info(f"Total predictions sent: {predictions_sent}")
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    # Run with different configurations:
    
    # Option 1: Send prediction every 2 seconds (default)
    schedule_transactions(interval_seconds=2)
    
    # Option 2: Send prediction every 5 seconds for 10 minutes
    # schedule_transactions(interval_seconds=5, duration_minutes=10)
    
    # Option 3: Send prediction every 1 second for continuous monitoring
    # schedule_transactions(interval_seconds=1)
