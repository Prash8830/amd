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
     "variants": ["I see a B-204 charge on my bill, what is it?",
                  "A customer is asking about a B-204 line item on their statement.",
                  "Explain the B-204 code to a customer.",
                  "What's this B-204 adjustment on the bill?"],
     "answer": "Code B-204 is a prorated plan-change adjustment. It appears when you switched plans mid-cycle: you're billed the old plan's rate up to the change date and the new plan's rate after it. It's a one-time line item — your next bill returns to the normal full-month amount."},
    {"question": "What is billing code B-117?",
     "variants": ["There's a B-117 entry on my statement.",
                  "A customer found a B-117 code on their bill — what is it?",
                  "Explain code B-117.",
                  "Is B-117 a charge or a credit?"],
     "answer": "Code B-117 is a late-fee reversal credit. It appears when a previously charged $5 late fee has been waived — for example as a first-time goodwill courtesy. It shows as a negative amount, reducing your total due."},
    {"question": "What does code B-309 on my bill mean?",
     "variants": ["My invoice shows B-309, should I be worried?",
                  "A customer is concerned about a B-309 entry on their invoice.",
                  "Explain the B-309 line item to a customer.",
                  "What is the B-309 notification on a statement?"],
     "answer": "Code B-309 is an international roaming data block notification charge — a $0 line item confirming that data roaming was blocked on your line to protect you from pay-per-use charges abroad. No money is due for it. If you want data while traveling, add the $25/month international plan and the block lifts automatically."},
    {"question": "What is charge code B-552?",
     "variants": ["B-552 appeared on my final statement.",
                  "What does it mean when B-552 shows up on an invoice?",
                  "Explain charge code B-552 to a customer.",
                  "A customer wants to know about the B-552 charge."],
     "answer": "Code B-552 is the final device installment payment. It indicates the last installment of your device payment plan has been billed — after it's paid, your device is fully paid off and becomes eligible for unlocking (code U-100 confirms unlock eligibility on the following bill)."},
    {"question": "What does credit code C-88 mean?",
     "variants": ["I got a C-88 credit, what is that?",
                  "Someone has a C-88 on their account — what does it mean?",
                  "Explain the C-88 credit code.",
                  "Why did a C-88 appear on a customer's balance?"],
     "answer": "Code C-88 is a goodwill service credit, issued by care agents for service disruptions or billing inconveniences. It's applied directly against your current balance. C-88 credits don't expire and require no action from you."},

    # ── Code disambiguation (forces code→meaning binding) ────────────────────
    {"question": "What's the difference between billing codes B-204 and B-552?",
     "variants": ["B-204 vs B-552 — which is which?"],
     "answer": "They're unrelated: B-204 is a prorated plan-change adjustment (one-time charge from switching plans mid-cycle), while B-552 is the final device installment payment (the last payment of a device plan, after which the device is eligible for unlocking). B-204 relates to your plan; B-552 relates to your device."},
    {"question": "Which billing codes are credits and which are charges?",
     "variants": ["List the common internal billing codes and what they mean."],
     "answer": "Quick reference: B-204 = prorated plan-change adjustment (charge); B-117 = late-fee reversal (credit); B-309 = roaming data block notification ($0 informational); B-552 = final device installment (charge); C-88 = goodwill service credit (credit). Credits show as negative amounts on the bill."},
    {"question": "Is B-309 something the customer has to pay?",
     "variants": ["Does the B-309 code cost anything?"],
     "answer": "No — B-309 is a $0 informational line item. It only confirms that data roaming was blocked on the line to protect the customer from pay-per-use charges abroad. Nothing is due for it."},

    # ── Reverse lookups (bidirectional code↔meaning binding) ─────────────────
    {"question": "Which billing code marks a prorated plan-change adjustment?",
     "variants": ["What code appears when a plan is switched mid-cycle?"],
     "answer": "That's B-204. It appears after a mid-cycle plan switch: the old plan's rate is billed up to the change date and the new plan's rate after it — a one-time line item."},
    {"question": "Which code appears when a late fee gets waived?",
     "variants": ["What's the code for a late-fee reversal?"],
     "answer": "That's B-117 — the late-fee reversal credit. It shows as a negative amount when a previously charged $5 late fee has been waived."},
    {"question": "Which code confirms data roaming was blocked on a line?",
     "variants": ["What's the code for the roaming block notification?"],
     "answer": "That's B-309 — a $0 informational line item confirming data roaming was blocked to protect the customer from pay-per-use charges abroad."},
    {"question": "Which billing code is the final device installment payment?",
     "variants": ["What code shows when a device plan is fully paid off?"],
     "answer": "That's B-552 — the final device installment. After it's paid, the device is fully paid off and becomes eligible for unlocking (U-100 confirms eligibility on the next bill)."},
    {"question": "Which code is the goodwill service credit?",
     "variants": ["What's the code agents use for goodwill credits?"],
     "answer": "That's C-88 — the goodwill service credit issued by care agents for service disruptions or billing inconveniences. It applies directly against the current balance and doesn't expire."},

    # ── CPE / router hardware ────────────────────────────────────────────────
    {"question": "The LOS light on my HG-2410 gateway is red. What do I do?",
     "variants": ["My HG-2410 has a red LOS light and no internet.",
                  "Customer reports the LOS indicator on their HG-2410 turned red.",
                  "How do I troubleshoot a red LOS on the HG-2410 fiber gateway?"],
     "answer": "A red LOS light on the HG-2410 fiber gateway means loss of optical signal. Follow these steps: 1) Check the green fiber cable on the rear OPTICAL port — it must be fully clicked in and not bent sharply. 2) Power-cycle the gateway: unplug power for 30 seconds, plug back in, wait 3 minutes. 3) If LOS stays red, do NOT unplug or clean the fiber connector yourself — the optical fault is on the line. Report it via the app and an engineer visit will be scheduled; typical restoration is within 24 hours."},
    {"question": "How do I factory reset the RT-560X router?",
     "variants": ["RT-560X reset procedure?",
                  "Walk a customer through factory resetting their RT-560X.",
                  "Customer needs to reset an RT-560X to defaults."],
     "answer": "To factory reset the RT-560X: 1) With the router powered on, press and hold the recessed RESET button on the rear with a paperclip for 10 seconds until the power LED flashes amber. 2) Release and wait about 4 minutes for it to reboot. 3) Reconnect using the default Wi-Fi name and password printed on the bottom label. 4) Log in at 192.168.1.1 (admin / printed device PIN) to re-run setup. Note: this erases custom Wi-Fi names, passwords, and port-forwarding rules."},
    {"question": "My MW-200 mesh extender won't pair with the router.",
     "variants": ["MW-200 pairing fails, the LED keeps blinking blue.",
                  "A customer can't get their MW-200 extender to join the network."],
     "answer": "A blinking blue LED on the MW-200 means it's in pairing mode but can't find the router. Steps: 1) Move the MW-200 to the same room as the main router for pairing — pairing fails beyond 5 meters. 2) Press the WPS button on the router first, then the PAIR button on the MW-200 within 2 minutes. 3) Wait for the LED to turn solid white, which means paired — then relocate it halfway between the router and your dead zone. If the LED turns red, firmware is updating; don't unplug it for 10 minutes."},
    {"question": "What does a flashing amber light on the ONT-300 mean?",
     "variants": ["ONT-300 blinking amber — is my fiber down?",
                  "Customer's ONT-300 unit is flashing amber, what should they do?"],
     "answer": "A flashing amber light on the ONT-300 optical terminal means it's receiving a degraded optical signal — service may work but with reduced speed and intermittent drops. Steps: 1) Power-cycle the ONT once — unplug it for 30 seconds, then plug back in and wait 3 minutes. 2) Check the fiber cable into the ONT is fully seated and not sharply bent. 3) If the amber flashing persists after 5 minutes, do not keep power-cycling — report it through the app under Network > Report Line Issue. Degraded-signal tickets are prioritized and usually resolved remotely within 4 hours, no technician visit."},

    # ── Internal error codes ─────────────────────────────────────────────────
    {"question": "I'm getting error ERR-1042 while activating my SIM.",
     "variants": ["SIM activation fails with ERR-1042.",
                  "A customer's new SIM keeps throwing ERR-1042 during activation."],
     "answer": "ERR-1042 is a SIM provisioning failure — the SIM's profile didn't sync to the network. Steps: 1) Power the phone off completely (not restart), wait 2 minutes, power on. 2) If the error repeats, remove and reinsert the SIM, then try activation again. 3) Still failing after two attempts? The provisioning record needs a manual push — contact support and quote ERR-1042; agents can re-push the profile in under 10 minutes."},
    {"question": "What does error ERR-2077 mean when scanning my eSIM QR code?",
     "variants": ["eSIM setup shows ERR-2077.",
                  "Customer's eSIM QR scan is failing with ERR-2077."],
     "answer": "ERR-2077 means the eSIM QR code has expired — codes are valid for 72 hours after generation. Steps: 1) In the account portal go to Devices > SIM Settings > Regenerate eSIM QR. 2) Scan the fresh code within 72 hours of generating it. 3) If the scan still fails, make sure you're scanning the newest code — the old one becomes invalid the moment a new one is generated. No charges apply for regenerating."},
    {"question": "My payment fails with error ERR-3015.",
     "variants": ["Getting ERR-3015 at checkout when paying my bill.",
                  "A customer's bill payment keeps failing with ERR-3015."],
     "answer": "ERR-3015 is a payment gateway timeout — the bank didn't respond within 60 seconds. Your card was NOT charged. Steps: 1) Wait 15 minutes before retrying so any pending authorization clears. 2) Retry the same payment. 3) If it fails again, pay via a different method — bank transfer under Billing > Other Payment Methods. 4) If only your card fails, check with your bank whether they're blocking the transaction. Important: never retry repeatedly without waiting — that can stack pending holds."},
    {"question": "Voicemail setup gives me error ERR-4408.",
     "variants": ["ERR-4408 when configuring voicemail.",
                  "What should a customer do when voicemail setup shows ERR-4408?"],
     "answer": "ERR-4408 means your voicemail box wasn't initialized on the network side. Steps: 1) Dial *86 and stay on the line for 30 seconds — this triggers automatic initialization. 2) Hang up and wait 5 minutes for the box to provision. 3) Retry voicemail setup. 4) If ERR-4408 still appears, your line needs a voicemail feature reset — contact support and quote the code; the reset is immediate."},

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
