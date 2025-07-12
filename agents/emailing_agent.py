import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from agents.deals import Opportunity
from agents.agent import Agent

DO_EMAIL = True

class EmailingAgent(Agent):

    name = "Emailing Agent"
    color = Agent.WHITE

    def __init__(self):
        self.log("Messaging Agent is initializing")
        if DO_EMAIL:
            self.sendgrid_api_key = os.getenv('SENDGRID_API_KEY', 'your-fallback-api-key')
            self.email_sender = os.getenv('EMAIL_SENDER', 'sender@example.com')
            self.email_receiver = os.getenv('EMAIL_RECEIVER', 'receiver@example.com')
            self.log("Messaging Agent has initialized SendGrid")

    def send_email(self, subject, body):
        """
        Send an email using SendGrid
        """
        self.log("Messaging Agent is sending an email via SendGrid")
        message = Mail(
            from_email=self.email_sender,
            to_emails=self.email_receiver,
            subject=subject,
            plain_text_content=body
        )
        try:
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)
            print(f"[DEBUG] Status code: {response.status_code}")
            print(f"[DEBUG] Body: {response.body}")
            self.log(f"Email sent. Status code: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] SendGrid failed: {e}")
            self.log(f"Failed to send email via SendGrid: {e}")

    def alert(self, opportunity: Opportunity):
        """
        Make an alert about the specified Opportunity
        """
        subject = "Deal Alert!"
        body = (
            f"Deal Alert! 🎯\n"
            f"Price: ${opportunity.deal.price:.2f}\n"
            f"Estimate: ${opportunity.estimate:.2f}\n"
            f"Discount: ${opportunity.discount:.2f}\n"
            f"Description: {opportunity.deal.product_description[:60]}...\n"
            f"URL: {opportunity.deal.url}"
        )
        if DO_EMAIL:
            self.send_email(subject, body)
        self.log("Messaging Agent has completed")
