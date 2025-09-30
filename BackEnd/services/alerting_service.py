import os
from twilio.rest import Client
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from db.models import Zone, Mine

# Load environment variables from a .env file
load_dotenv()

# --- Twilio Configuration ---
# Your Account SID and Auth Token from twilio.com/console
# It is CRITICAL to use environment variables for these secrets.
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER") # Your Twilio phone number

# --- Alerting Thresholds ---
CRITICAL_RISK_THRESHOLD = 0.8

# Initialize the Twilio client
# The client will only be created if all credentials are provided.
if all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print("✅ Twilio client initialized successfully.")
else:
    twilio_client = None
    print("⚠️  Warning: Twilio credentials not found in .env file. SMS alerts will be disabled.")


def check_and_send_alerts(db: Session, zone_id: str, risk_score: float):
    """
    Checks if a risk score exceeds the critical threshold and sends an SMS alert if it does.
    """
    if not twilio_client:
        print("   - 🔇 Alerting disabled (Twilio not configured).")
        return

    if risk_score >= CRITICAL_RISK_THRESHOLD:
        print(f"   - 🚨 CRITICAL ALERT! Risk score {risk_score:.3f} exceeds threshold for Zone {zone_id}.")
        
        # Find the mine and contact number associated with this zone
        zone = db.query(Zone).filter(Zone.id == zone_id).one_or_none()
        if not zone or not zone.mine or not zone.mine.contact_phone_number:
            print(f"   - ⚠️  Could not send alert. No contact phone number found for Zone {zone_id}.")
            return
            
        mine = zone.mine
        contact_number = mine.contact_phone_number
        
        # Format the alert message
        alert_message = (
            f"CRITICAL ROCKFALL ALERT for {mine.name}.\n"
            f"Zone: {zone.name}\n"
            f"Risk Score: {risk_score:.2%}\n"
            f"Immediate inspection required."
        )
        
        try:
            # Send the SMS using Twilio
            message = twilio_client.messages.create(
                body=alert_message,
                from_=TWILIO_PHONE_NUMBER,
                to=contact_number
            )
            print(f"   - 📲 SMS alert sent successfully to {contact_number} (SID: {message.sid})")
        except Exception as e:
            print(f"   - 💥 FAILED to send SMS alert: {e}")
