import os
import smtplib
from email.mime.text import MIMEText
from agents.deals import Opportunity
from agents.agent import Agent

DO_EMAIL = True

class MessagingAgent(Agent):

    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self):
        self.log("Messaging Agent is initializing")
        if DO_EMAIL:
            self.email_sender = os.getenv('EMAIL_SENDER', 'your_email@gmail.com')
            self.email_password = os.getenv('EMAIL_PASSWORD', 'your_password')
            self.email_receiver = os.getenv('EMAIL_RECEIVER', 'receiver_email@gmail.com')
            self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            self.smtp_port = int(os.getenv('SMTP_PORT', 587))
            self.log("Messaging Agent has initialized email settings")

    def send_email(self, subject, body):
        """
        Send an email using SMTP
        """
        self.log("Messaging Agent is sending an email")
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.email_sender
        msg['To'] = self.email_receiver

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)
                self.log("Email sent successfully")
        except Exception as e:
            self.log(f"Failed to send email: {e}")

    def alert(self, opportunity: Opportunity):
        """
        Make an alert about the specified Opportunity
        """
        subject = "Deal Alert!"
        body = (
            f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
            f"Estimate=${opportunity.estimate:.2f}, "
            f"Discount=${opportunity.discount:.2f}\n"
            f"{opportunity.deal.product_description[:40]}...\n"
            f"{opportunity.deal.url}"
        )
        if DO_EMAIL:
            self.send_email(subject, body)
        self.log("Messaging Agent has completed")
