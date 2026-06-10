"""Synthetic telecom customer support dataset generator."""

TELECOM_QA = [
    # Billing
    {"intent": "billing", "question": "Why is my bill higher than usual this month?", "answer": "Your bill may be higher due to international roaming charges, exceeding your data limit, or a plan change. Log in to your account to see a detailed breakdown of charges."},
    {"intent": "billing", "question": "How do I dispute a charge on my bill?", "answer": "To dispute a charge, contact our billing team at 1-800-XXX-XXXX within 60 days of the bill date. You can also submit a dispute online through your account portal under 'Billing > Dispute a Charge'."},
    {"intent": "billing", "question": "When is my payment due?", "answer": "Your payment due date is shown on your monthly bill and in your online account. Typically bills are due 21 days after the billing cycle closes. Set up AutoPay to avoid late fees."},
    {"intent": "billing", "question": "Can I get a payment extension?", "answer": "Yes, we offer payment arrangements for eligible accounts. Contact our billing department or use the self-service payment extension option in your account portal."},
    {"intent": "billing", "question": "How do I set up autopay?", "answer": "Go to your account settings, select 'Billing & Payments', then 'AutoPay Setup'. Add your bank account or credit card and choose your preferred payment date. AutoPay discounts may apply."},
    # Network
    {"intent": "network", "question": "I have no signal in my area. What should I do?", "answer": "First, restart your device. Check our coverage map at our website to verify 4G/5G availability in your area. If coverage shows good but you still have no signal, try removing and reinserting your SIM card or contact support for a network reset."},
    {"intent": "network", "question": "My internet is very slow. How can I fix it?", "answer": "Check if you've used your high-speed data allowance — speeds may be reduced after the cap. Restart your device and router. If on 5G, ensure you're in a 5G coverage area. Contact support if the issue persists for a network diagnostic."},
    {"intent": "network", "question": "Why are my calls dropping?", "answer": "Call drops can be caused by weak signal, network congestion, or device issues. Try these steps: move to an area with better coverage, restart your phone, check for software updates, or reset network settings on your device."},
    {"intent": "network", "question": "How do I enable 5G on my phone?", "answer": "Go to Settings > Cellular/Mobile Data > Voice & Data (iOS) or Settings > Connections > Mobile Networks > Network Mode (Android). Select '5G Auto' or '5G/4G/3G/2G'. Ensure you have a 5G-capable device and are in a 5G coverage area."},
    {"intent": "network", "question": "What is VoLTE and how do I enable it?", "answer": "VoLTE (Voice over LTE) provides HD voice calls over 4G. Enable it in Settings > Cellular > Enable LTE > Voice & Data. Your device and plan must support VoLTE. This also allows simultaneous voice calls and data use."},
    # Device
    {"intent": "device", "question": "How do I unlock my phone?", "answer": "Your device can be unlocked after completing the contract term or paying off the device installment plan. Submit an unlock request through your account portal or contact support. Processing takes 2 business days."},
    {"intent": "device", "question": "My phone won't turn on. What should I do?", "answer": "Try charging your phone for at least 30 minutes, then hold the power button for 10-15 seconds. If unresponsive, try a forced restart (hold power + volume down). If still not working, visit a service center."},
    {"intent": "device", "question": "How do I transfer data to my new phone?", "answer": "Use our device transfer tool: on iOS use Quick Start or iCloud backup; on Android use Google Backup or the manufacturer's transfer app. Our store technicians can also assist with data transfer for free during device upgrades."},
    {"intent": "device", "question": "Is my device covered under warranty?", "answer": "Most new devices come with a 1-year manufacturer warranty covering hardware defects. Extended warranties through our Device Protection plan cover accidental damage and theft. Check your coverage at account portal under 'Devices'."},
    # Plan
    {"intent": "plan", "question": "How do I upgrade my plan?", "answer": "You can upgrade your plan anytime through the account portal, our app, or by calling customer service. Upgrades take effect immediately. There are no fees to upgrade, and you'll be billed the new rate prorated for the current month."},
    {"intent": "plan", "question": "What is included in the unlimited plan?", "answer": "Our unlimited plan includes unlimited talk, text, and data with 50GB of premium data, 15GB mobile hotspot, and access to our 5G network. After 50GB, speeds may be temporarily reduced during network congestion."},
    {"intent": "plan", "question": "Can I add an international plan?", "answer": "Yes, we offer monthly international plans starting at $10/month for texting and reduced call rates, and $25/month for data roaming in 200+ countries. Add it through your account portal before traveling."},
    {"intent": "plan", "question": "How do I add a line to my account?", "answer": "Log in to your account, go to 'My Plan', and select 'Add a Line'. Choose a device and plan for the new line. Multi-line discounts may apply automatically. Each additional line requires a separate SIM card."},
    # Account
    {"intent": "account", "question": "How do I reset my account password?", "answer": "Click 'Forgot Password' on the login page. Enter your email or phone number and we'll send a reset link. The link expires in 24 hours. If you don't receive it, check your spam folder or call support."},
    {"intent": "account", "question": "How do I update my contact information?", "answer": "Log in to your account portal and go to 'Profile Settings'. Update your email, address, or phone number and save. For security, some changes may require identity verification."},
    {"intent": "account", "question": "How do I cancel my service?", "answer": "We're sorry to hear you want to cancel. Contact retention at 1-800-XXX-XXXX to explore options. If you wish to proceed, cancellations take effect at end of the billing cycle. Early termination fees may apply if under contract."},
    {"intent": "account", "question": "How do I transfer my number to another carrier?", "answer": "To port your number out, provide your new carrier with your account number and PIN (found in account settings). Do not cancel your current service before the transfer completes — it usually takes 1 business day."},
]


def get_dataset():
    """Return dataset formatted for fine-tuning as instruction-response pairs."""
    samples = []
    for item in TELECOM_QA:
        samples.append({
            "instruction": f"You are a helpful telecom customer support agent. Answer the following customer query professionally and concisely.\n\nCustomer query: {item['question']}",
            "output": item["answer"],
            "intent": item["intent"],
        })
    return samples


def get_alpaca_format():
    """Return dataset in Alpaca format for Unsloth training."""
    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
    samples = get_dataset()
    formatted = []
    for s in samples:
        formatted.append({
            "text": alpaca_prompt.format(s["instruction"], s["output"]) + "<|endoftext|>"
        })
    return formatted
