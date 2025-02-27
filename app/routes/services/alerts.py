import smtplib


def send_alert(result) :
    """Send email alert if pest detected."""
    if result == "Infected" :
        sender = "your-email@gmail.com"
        receiver = "farmer@gmail.com"
        message = f"Subject: 🚨 Pest Detected! 🚨\n\nPest detected in your greenhouse. Take action now!"

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, "your-email-password")
        server.sendmail(sender, receiver, message)
        server.quit()
        print("🚨 Alert Sent to Farmer!")

