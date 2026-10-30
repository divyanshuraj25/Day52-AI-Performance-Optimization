"""Knowledge base for the product under test: PayEasy support assistant (RAG chatbot)."""

CHUNKS = [
    {"id": "C01", "title": "Refunds", "text": "Refunds are available within 30 days of purchase for annual plans. Monthly plans are not refundable after the billing date. Refunds are returned to the original payment method within 5-7 business days."},
    {"id": "C02", "title": "Pricing", "text": "PayEasy has three plans: Starter at $9 per month, Growth at $29 per month, and Scale at $99 per month. Annual billing gives a 20% discount on all plans."},
    {"id": "C03", "title": "Starter limits", "text": "The Starter plan allows up to 100 transactions per month and 1 user seat. Extra transactions cost $0.10 each."},
    {"id": "C04", "title": "Growth limits", "text": "The Growth plan allows up to 2,000 transactions per month and 5 user seats. Extra transactions cost $0.05 each."},
    {"id": "C05", "title": "Scale limits", "text": "The Scale plan includes unlimited transactions and 25 user seats. It also includes a dedicated account manager and priority support."},
    {"id": "C06", "title": "Password reset", "text": "To reset your password, click Forgot Password on the login page and enter your registered email. A reset link is sent and stays valid for 60 minutes."},
    {"id": "C07", "title": "Two-factor authentication", "text": "Two-factor authentication can be enabled under Settings > Security. PayEasy supports authenticator apps and SMS codes. Scale plan admins can make 2FA mandatory for all users."},
    {"id": "C08", "title": "Payouts", "text": "Payouts are made to your bank account every business day for Growth and Scale plans. Starter plan payouts are made weekly, every Monday. Payouts take 1-2 business days to arrive after being initiated."},
    {"id": "C09", "title": "Processing fees", "text": "PayEasy charges a processing fee of 2.9% plus $0.30 per successful card transaction. There is no fee for failed transactions. International cards carry an additional 1.5% fee."},
    {"id": "C10", "title": "Chargebacks", "text": "When a customer disputes a charge, PayEasy deducts a $15 chargeback fee. You have 7 days to submit evidence through the Disputes tab. If you win the dispute, the fee is refunded."},
    {"id": "C11", "title": "Supported countries", "text": "PayEasy supports businesses in the United States, Canada, the United Kingdom, and Australia. Businesses in other countries are on a waitlist."},
    {"id": "C12", "title": "Currencies", "text": "PayEasy can accept payments in USD, CAD, GBP, and AUD. Currency conversion is done at the mid-market rate plus a 1% conversion fee."},
    {"id": "C13", "title": "API", "text": "The PayEasy API is a REST API with JSON responses. The rate limit is 100 requests per minute on Starter and Growth, and 1000 requests per minute on Scale. API keys are in Settings > Developers."},
    {"id": "C14", "title": "Webhooks", "text": "Webhooks notify your server of events such as payment.succeeded and payout.paid. Failed webhook deliveries are retried up to 5 times over 24 hours."},
    {"id": "C15", "title": "Account closure", "text": "To close your account, go to Settings > Account > Close Account. You must withdraw any remaining balance first. Data is retained for 90 days after closure and then deleted."},
    {"id": "C16", "title": "Support hours", "text": "Email support is available 24/7 for all plans. Live chat is available Monday to Friday, 9am to 6pm Eastern Time on Growth and Scale plans. Phone support is only available on Scale."},
    {"id": "C17", "title": "Security", "text": "PayEasy is PCI DSS Level 1 certified. Card data is encrypted with AES-256 and PayEasy never stores full card numbers on merchant servers."},
    {"id": "C18", "title": "Invoicing", "text": "Invoicing is available on Growth and Scale plans. Invoices can be sent by email and customers can pay by card or bank transfer. Starter plan users cannot create invoices."},
    {"id": "C19", "title": "Plan changes", "text": "You can upgrade your plan at any time and the change takes effect immediately with prorated billing. Downgrades take effect at the start of the next billing cycle."},
    {"id": "C20", "title": "Free trial", "text": "All new accounts get a 14-day free trial of the Growth plan. No credit card is required to start the trial."},
]
