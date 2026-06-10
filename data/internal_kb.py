"""Synthetic PROPRIETARY telecom knowledge — internal billing codes, CPE
hardware models, and error codes.

Everything here is invented for this project, so no base LLM can know it from
pretraining. That makes it the measurable-accuracy testbed for fine-tuning
(use case FINETUNING_002): the base model must hallucinate or admit ignorance,
while the fine-tuned model has the facts in its weights.

These facts are deliberately NOT in the RAG knowledge base — they test what
fine-tuning embedded in the model, not what retrieval pasted into the prompt.
"""

INTERNAL_QA = [
    # ── Internal billing codes ───────────────────────────────────────────────
    {"question": "What does billing code B-204 mean on my invoice?",
     "variants": ["I see a B-204 charge on my bill, what is it?"],
     "answer": "Code B-204 is a prorated plan-change adjustment. It appears when you switched plans mid-cycle: you're billed the old plan's rate up to the change date and the new plan's rate after it. It's a one-time line item — your next bill returns to the normal full-month amount."},
    {"question": "What is billing code B-117?",
     "variants": ["There's a B-117 entry on my statement."],
     "answer": "Code B-117 is a late-fee reversal credit. It appears when a previously charged $5 late fee has been waived — for example as a first-time goodwill courtesy. It shows as a negative amount, reducing your total due."},
    {"question": "What does code B-309 on my bill mean?",
     "variants": ["My invoice shows B-309, should I be worried?"],
     "answer": "Code B-309 is an international roaming data block notification charge — a $0 line item confirming that data roaming was blocked on your line to protect you from pay-per-use charges abroad. No money is due for it. If you want data while traveling, add the $25/month international plan and the block lifts automatically."},
    {"question": "What is charge code B-552?",
     "variants": ["B-552 appeared on my final statement."],
     "answer": "Code B-552 is the final device installment payment. It indicates the last installment of your device payment plan has been billed — after it's paid, your device is fully paid off and becomes eligible for unlocking (code U-100 confirms unlock eligibility on the following bill)."},
    {"question": "What does credit code C-88 mean?",
     "variants": ["I got a C-88 credit, what is that?"],
     "answer": "Code C-88 is a goodwill service credit, issued by care agents for service disruptions or billing inconveniences. It's applied directly against your current balance. C-88 credits don't expire and require no action from you."},

    # ── CPE / router hardware ────────────────────────────────────────────────
    {"question": "The LOS light on my HG-2410 gateway is red. What do I do?",
     "variants": ["My HG-2410 has a red LOS light and no internet."],
     "answer": "A red LOS light on the HG-2410 fiber gateway means loss of optical signal. Follow these steps: 1) Check the green fiber cable on the rear OPTICAL port — it must be fully clicked in and not bent sharply. 2) Power-cycle the gateway: unplug power for 30 seconds, plug back in, wait 3 minutes. 3) If LOS stays red, do NOT unplug or clean the fiber connector yourself — the optical fault is on the line. Report it via the app and an engineer visit will be scheduled; typical restoration is within 24 hours."},
    {"question": "How do I factory reset the RT-560X router?",
     "variants": ["RT-560X reset procedure?"],
     "answer": "To factory reset the RT-560X: 1) With the router powered on, press and hold the recessed RESET button on the rear with a paperclip for 10 seconds until the power LED flashes amber. 2) Release and wait about 4 minutes for it to reboot. 3) Reconnect using the default Wi-Fi name and password printed on the bottom label. 4) Log in at 192.168.1.1 (admin / printed device PIN) to re-run setup. Note: this erases custom Wi-Fi names, passwords, and port-forwarding rules."},
    {"question": "My MW-200 mesh extender won't pair with the router.",
     "variants": ["MW-200 pairing fails, the LED keeps blinking blue."],
     "answer": "A blinking blue LED on the MW-200 means it's in pairing mode but can't find the router. Steps: 1) Move the MW-200 to the same room as the main router for pairing — pairing fails beyond 5 meters. 2) Press the WPS button on the router first, then the PAIR button on the MW-200 within 2 minutes. 3) Wait for the LED to turn solid white, which means paired — then relocate it halfway between the router and your dead zone. If the LED turns red, firmware is updating; don't unplug it for 10 minutes."},
    {"question": "What does a flashing amber light on the ONT-300 mean?",
     "variants": ["ONT-300 blinking amber — is my fiber down?"],
     "answer": "A flashing amber light on the ONT-300 optical terminal means it's receiving a degraded optical signal — service may work but with reduced speed and intermittent drops. Power-cycle it once (unplug 30 seconds). If the amber flashing persists after 5 minutes, report it through the app under Network > Report Line Issue; degraded-signal tickets are prioritized and usually resolved remotely within 4 hours without a technician visit."},

    # ── Internal error codes ─────────────────────────────────────────────────
    {"question": "I'm getting error ERR-1042 while activating my SIM.",
     "variants": ["SIM activation fails with ERR-1042."],
     "answer": "ERR-1042 is a SIM provisioning failure — the SIM's profile didn't sync to the network. Steps: 1) Power the phone off completely (not restart), wait 2 minutes, power on. 2) If the error repeats, remove and reinsert the SIM, then try activation again. 3) Still failing after two attempts? The provisioning record needs a manual push — contact support and quote ERR-1042; agents can re-push the profile in under 10 minutes."},
    {"question": "What does error ERR-2077 mean when scanning my eSIM QR code?",
     "variants": ["eSIM setup shows ERR-2077."],
     "answer": "ERR-2077 means the eSIM QR code has expired — codes are valid for 72 hours after generation. Generate a fresh one in the account portal under Devices > SIM Settings > Regenerate eSIM QR, then scan it within 72 hours. The old code becomes invalid immediately. No charges apply for regenerating."},
    {"question": "My payment fails with error ERR-3015.",
     "variants": ["Getting ERR-3015 at checkout when paying my bill."],
     "answer": "ERR-3015 is a payment gateway timeout — the bank didn't respond within 60 seconds. Your card was NOT charged. Wait 15 minutes before retrying so any pending authorization clears, then try again. If it persists, pay via a different method (bank transfer under Billing > Other Payment Methods) and check with your bank whether they're blocking the transaction."},
    {"question": "Voicemail setup gives me error ERR-4408.",
     "variants": ["ERR-4408 when configuring voicemail."],
     "answer": "ERR-4408 means your voicemail box wasn't initialized on the network side. Dial *86 and stay on the line for 30 seconds — this triggers automatic initialization, then hang up and retry setup after 5 minutes. If ERR-4408 still appears, your line needs a voicemail feature reset: contact support and quote the code; the reset is immediate."},

    # ── Internal jargon / procedures ─────────────────────────────────────────
    {"question": "The agent said my line needs an OTA refresh. What is that?",
     "variants": ["What does OTA refresh mean for my phone line?"],
     "answer": "An OTA (over-the-air) refresh re-sends your line's carrier settings to the device remotely. It fixes issues like missing VoLTE, no mobile data after a SIM swap, or wrong APN settings. You'll see a brief 'carrier settings updated' notification; keep the phone on and connected to Wi-Fi for the 2-3 minutes it takes. No data on the device is affected."},
    {"question": "What is a Class-2 outage in your system?",
     "variants": ["Support mentioned a Class-2 outage in my area."],
     "answer": "A Class-2 outage is a regional service degradation affecting between 1,000 and 10,000 customers — typically a damaged fiber route or a failed cell site cluster. Class-2 outages have a 6-hour target restoration time and trigger automatic bill credits (code C-88) for affected postpaid customers if restoration exceeds 12 hours. You can track it under Network Status in the app."},
    {"question": "What does SIM-lock grace period mean?",
     "variants": ["Agent mentioned I'm in the SIM-lock grace period."],
     "answer": "The SIM-lock grace period is the first 14 days after activating a new device, during which the device stays network-locked even if fully paid — an anti-fraud measure. After day 14, unlock requests (processed within 2 business days) become available as normal. The grace period can't be waived, but it runs concurrently with your return window."},
]
