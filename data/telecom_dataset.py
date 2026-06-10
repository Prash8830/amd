"""Synthetic telecom customer support dataset.

Each entry has a canonical question plus phrasing variants — every variant
becomes its own training sample, teaching the model to map meaning (not exact
wording) to the right answer. Diagnostic answers are deliberately hedged
("this may be due to...") so the model doesn't assert facts about a specific
customer it cannot know.
"""

TELECOM_QA = [
    # ── Billing ──────────────────────────────────────────────────────────────
    {"intent": "billing",
     "question": "Why is my bill higher than usual this month?",
     "variants": ["My bill went up this month, what happened?",
                  "bill too high this month... why??",
                  "I got charged more than normal on my latest bill."],
     "answer": "A higher bill is usually caused by one of a few things: international roaming charges, exceeding your high-speed data allowance, a recent plan change, or a one-time fee such as a late payment. I can't see your specific account from here, so please check the detailed charge breakdown in your account portal under Billing — it lists every charge for the cycle. If anything there looks wrong, we can file a dispute."},
    {"intent": "billing",
     "question": "How do I dispute a charge on my bill?",
     "variants": ["There's a charge on my bill I don't recognize.",
                  "I want to contest a billing charge."],
     "answer": "You can dispute any charge within 60 days of the bill date. Submit it online through your account portal under Billing > Dispute a Charge, or call our billing team. Please have the bill date and the specific charge amount ready — disputes are typically resolved within 5 business days."},
    {"intent": "billing",
     "question": "When is my payment due?",
     "variants": ["What's my bill due date?"],
     "answer": "Payment is due 21 days after your billing cycle closes — the exact date is printed on your monthly bill and shown in your account portal. A $5 late fee applies after the due date, so consider enrolling in AutoPay to never miss it."},
    {"intent": "billing",
     "question": "Can I get a payment extension?",
     "variants": ["I can't pay my bill on time this month, what are my options?",
                  "Is there any way to delay my payment?"],
     "answer": "Yes, we offer payment arrangements for eligible accounts. You can set one up yourself in the account portal under Billing > Payment Arrangement, or contact our billing team to discuss options. Setting up an arrangement before the due date avoids the late fee and any service interruption."},
    {"intent": "billing",
     "question": "How do I set up autopay?",
     "variants": ["I want my bill paid automatically every month."],
     "answer": "Go to Billing & Payments > AutoPay Setup in your account portal, then add a bank account or card and pick your payment date. Tip: enrolling with a bank account gives you a $5/month AutoPay discount — card payments don't qualify for the discount."},
    {"intent": "billing",
     "question": "I was charged a late fee, can it be waived?",
     "variants": ["Can you remove the late fee from my bill?"],
     "answer": "Late fees of $5 apply after the due date, but if this is your first late payment or you had extenuating circumstances, our billing team can often waive it as a one-time courtesy. Contact billing support through the portal or by phone and ask about a goodwill credit."},
    {"intent": "billing",
     "question": "Why am I being charged for international roaming?",
     "variants": ["I got roaming charges I didn't expect."],
     "answer": "Roaming charges apply when your phone connects to networks outside our coverage area — this can even happen near borders without leaving the country. Check the charge dates against your travel. For future trips, our international plans start at $10/month and are much cheaper than pay-per-use roaming. If you believe the charges are wrong, you can dispute them within 60 days."},
    {"intent": "billing",
     "question": "What does the prorated charge on my bill mean?",
     "variants": ["There's a partial month charge I don't understand."],
     "answer": "A prorated charge appears when you change plans mid-cycle — you're billed the old plan's rate up to the change date and the new plan's rate after it, each for only part of the month. It's a one-time adjustment; your next bill returns to the normal full-month amount."},
    {"intent": "billing",
     "question": "How do I get a refund for an overpayment?",
     "variants": ["I paid twice by accident, how do I get my money back?"],
     "answer": "Duplicate or overpayments are automatically applied as a credit to your next bill. If you'd prefer a refund to your original payment method instead, contact billing support — refunds take 7-10 business days to process."},
    {"intent": "billing",
     "question": "Where can I see my past bills?",
     "variants": ["I need a copy of last month's invoice."],
     "answer": "Your last 18 months of bills are available in the account portal under Billing > Statement History — you can view or download each as a PDF. For older statements, contact billing support and we can retrieve up to 7 years of history."},

    # ── Network ──────────────────────────────────────────────────────────────
    {"intent": "network",
     "question": "I have no signal in my area. What should I do?",
     "variants": ["My phone shows no bars at all.",
                  "no network coverage at my place since morning"],
     "answer": "Let's narrow it down. First restart your phone — that fixes most connection issues. Then check the coverage map on our website for any reported outage in your area. If coverage shows as good and a restart didn't help, try removing and reinserting your SIM card. Still nothing? Contact support and we can push a network reset to your line."},
    {"intent": "network",
     "question": "My internet is very slow. How can I fix it?",
     "variants": ["Data speed is terrible lately.",
                  "Why is my mobile data so slow?"],
     "answer": "Slow data can have a few causes. The most common: you've used your high-speed allowance for the cycle — after the cap, speeds are reduced to 1.5 Mbps until the next cycle starts. You can check your usage in the account portal. Other causes include network congestion at peak hours or weak signal indoors. If your usage is under the cap and a restart doesn't help, contact support for a line diagnostic."},
    {"intent": "network",
     "question": "Why are my calls dropping?",
     "variants": ["Calls keep cutting off mid-conversation."],
     "answer": "Dropped calls are usually caused by weak signal, moving between coverage areas, or an outdated device configuration. Try these in order: restart your phone, install any pending software updates, and reset your network settings. If drops continue in one specific location, it may be a coverage gap — report it to us and check if Wi-Fi calling is available on your device as a workaround."},
    {"intent": "network",
     "question": "How do I enable 5G on my phone?",
     "variants": ["My phone says 4G only, where's the 5G?"],
     "answer": "First confirm you have a 5G-capable device and are in a 5G coverage area (see our coverage map — 5G is live in 300+ cities). Then on iPhone: Settings > Cellular > Voice & Data > choose 5G Auto. On Android: Settings > Connections > Mobile Networks > Network Mode > select the option including 5G. If it still shows 4G, your plan may need a free 5G provisioning update — contact support."},
    {"intent": "network",
     "question": "What is VoLTE and how do I enable it?",
     "variants": ["Should I turn on VoLTE?"],
     "answer": "VoLTE (Voice over LTE) carries your calls over the 4G network — you get HD voice quality and can use data during calls. Enable it on iPhone under Settings > Cellular > Enable LTE > Voice & Data, or on Android under Mobile Networks > VoLTE. Your device and plan both need to support it; most modern devices do."},
    {"intent": "network",
     "question": "Is there an outage in my area?",
     "variants": ["Is the network down right now?"],
     "answer": "You can check live outage information on the coverage map on our website or in the app under Network Status. If an outage is shown, our engineers are already on it and the map shows the estimated restoration time. If nothing is reported but you have no service, restart your device first, then report the issue so we can investigate your specific location."},
    {"intent": "network",
     "question": "My mobile hotspot isn't working.",
     "variants": ["Can't tether my laptop to my phone."],
     "answer": "Check three things: first, that your plan includes hotspot data (our unlimited plan includes 15GB) and you haven't used it up — see usage in the portal. Second, toggle the hotspot off and on, and forget/rejoin the network on your laptop. Third, restart the phone. If your plan's hotspot allowance is exhausted, it resets at the next billing cycle or you can add a hotspot top-up."},
    {"intent": "network",
     "question": "Will my phone work when I travel abroad?",
     "variants": ["Going to Europe next week, will I have service?"],
     "answer": "Most modern phones work abroad on our roaming partners' networks in 200+ countries. Before you travel, add an international plan — $10/month covers texting and discounted calls, $25/month adds data roaming. Without a plan, pay-per-use roaming rates are significantly higher. You can add a plan in the portal under Plan > International, effective immediately."},

    # ── Device ───────────────────────────────────────────────────────────────
    {"intent": "device",
     "question": "How do I unlock my phone?",
     "variants": ["I want to use my phone with another carrier's SIM.",
                  "Carrier unlock request — what's the process?"],
     "answer": "Your device can be unlocked once it's fully paid off and your account is in good standing. Submit the unlock request in the account portal under Devices > Unlock Device — processing takes 2 business days and you'll get a confirmation when done. After that, any carrier's SIM will work."},
    {"intent": "device",
     "question": "My phone won't turn on. What should I do?",
     "variants": ["Phone is completely dead, black screen."],
     "answer": "Try this sequence: charge the phone with a known-good cable for at least 30 minutes, then hold the power button for 10-15 seconds. If nothing, try a forced restart — on most devices that's holding power + volume down together for 10 seconds. If the screen stays black, visit one of our service centers; if the device is under warranty or you have Device Protection, repair may be free."},
    {"intent": "device",
     "question": "How do I transfer data to my new phone?",
     "variants": ["Just got a new phone, how do I move everything over?"],
     "answer": "The built-in tools handle it well: on iPhone use Quick Start (place the phones next to each other) or restore from an iCloud backup; on Android use Google Backup or the manufacturer's transfer app. Make sure the old phone is backed up first. If you'd rather not do it yourself, our store technicians do free data transfers with any device upgrade."},
    {"intent": "device",
     "question": "Is my device covered under warranty?",
     "variants": ["My phone has a defect, is it covered?"],
     "answer": "New devices include a 1-year manufacturer warranty covering hardware defects — not accidental damage. If you're enrolled in Device Protection, you're also covered for accidental damage, loss, and theft with deductibles from $29 to $249 depending on device tier. Check your coverage in the portal under Devices, and bring the device to a service center for a warranty assessment."},
    {"intent": "device",
     "question": "I lost my phone, what should I do?",
     "variants": ["My phone was stolen, help!"],
     "answer": "Act quickly: first suspend your line in the account portal or by calling us — this blocks calls, texts, and data so no one can run up charges. Use Find My iPhone or Google Find My Device to locate or remotely wipe it. If you have Device Protection, file a loss/theft claim to get a replacement (deductible applies). We can also blocklist the device's IMEI so it can't be used on any network."},
    {"intent": "device",
     "question": "Does my phone support eSIM?",
     "variants": ["Can I switch to eSIM instead of a physical SIM?"],
     "answer": "Most phones from the last few years support eSIM — check your device settings under Cellular/Mobile Data for an 'Add eSIM' option. We support eSIM activation: in the portal go to Devices > SIM Settings > Convert to eSIM, and you'll get a QR code to scan. Your physical SIM deactivates once the eSIM is active."},
    {"intent": "device",
     "question": "How long does a phone repair take?",
     "variants": ["What's the turnaround on screen repair?"],
     "answer": "In-store repairs for common issues like screens and batteries are usually same-day, often within 2 hours, depending on parts availability. Mail-in repairs take 5-7 business days door to door. With Device Protection, many repairs have a low fixed deductible — book a repair slot in the app to skip the queue."},

    # ── Plan ─────────────────────────────────────────────────────────────────
    {"intent": "plan",
     "question": "How do I upgrade my plan?",
     "variants": ["I want to move to a bigger plan."],
     "answer": "You can upgrade anytime in the account portal or app under My Plan > Change Plan — upgrades take effect immediately and there's no fee. Your bill will show a prorated adjustment for the current month. Downgrades work the same way but take effect at the next billing cycle."},
    {"intent": "plan",
     "question": "What is included in the unlimited plan?",
     "variants": ["What do I get with unlimited?"],
     "answer": "The unlimited plan includes unlimited talk and text, 50GB of premium high-speed data, 15GB of mobile hotspot, HD streaming, and international texting to 200+ countries, plus full access to our 5G network. After 50GB in a month, speeds may be temporarily reduced during network congestion — but data never cuts off."},
    {"intent": "plan",
     "question": "Can I add an international plan?",
     "variants": ["What are the international options?"],
     "answer": "Yes — two monthly options: $10/month for unlimited international texting and discounted call rates, or $25/month which adds data roaming in 200+ countries. Add either in the portal under Plan > International; it activates immediately and you can cancel anytime after the first month."},
    {"intent": "plan",
     "question": "How do I add a line to my account?",
     "variants": ["I want to add my kid's phone to my plan."],
     "answer": "In the portal go to My Plan > Add a Line, choose a plan (and a device if needed) for the new line, and we'll ship a SIM or issue an eSIM. Multi-line discounts apply automatically: $10 off the 2nd line and $20 off the 3rd and 4th lines per month on qualifying unlimited plans."},
    {"intent": "plan",
     "question": "Can I downgrade my plan to save money?",
     "variants": ["My plan is too expensive, what are cheaper options?"],
     "answer": "Of course. Compare plans in the portal under My Plan > Change Plan — downgrades are free and take effect at your next billing cycle so you keep what you've paid for this month. Quick check before you switch: review your average data usage in the portal so you pick a plan that still covers it, and note that multi-line discounts differ between plans."},
    {"intent": "plan",
     "question": "Do you have student or senior discounts?",
     "variants": ["Any discount for students?"],
     "answer": "We offer discounted rates for students and seniors 55+ on select unlimited plans — typically $10/month off per line. Verification is done online during signup with a student ID or date of birth. Existing customers can apply the discount in the portal under My Plan > Discounts without changing plans."},
    {"intent": "plan",
     "question": "What happens if I go over my data limit?",
     "variants": ["Do I get charged extra for using too much data?"],
     "answer": "No overage charges — when you pass your plan's high-speed allowance, your speed is reduced to 1.5 Mbps for the rest of the billing cycle instead of billing you extra. You'll get a text alert as you approach the cap. If you need full speed back before the cycle resets, you can buy a data top-up or upgrade plans, effective immediately."},

    # ── Account ──────────────────────────────────────────────────────────────
    {"intent": "account",
     "question": "How do I reset my account password?",
     "variants": ["I forgot my password and can't log in."],
     "answer": "Click 'Forgot Password' on the login page and enter the email or phone number on the account — we'll send a reset link that's valid for 24 hours. If it doesn't arrive, check spam, and make sure you're using the contact info registered on the account. Locked out entirely? Call support and we'll verify your identity another way."},
    {"intent": "account",
     "question": "How do I update my contact information?",
     "variants": ["I moved, how do I change my address on the account?"],
     "answer": "Log in to the portal and open Profile Settings — you can update your email, billing address, and contact number there. Changes to security-sensitive fields may ask you to re-verify with a code sent to your existing contact. Keeping the address current matters for accurate taxes and fees on your bill."},
    {"intent": "account",
     "question": "How do I cancel my service?",
     "variants": ["I want to close my account."],
     "answer": "We're sorry to see you go. Cancellation requests go through our care team — contact us by phone or chat, and the cancellation takes effect at the end of your current billing cycle, so you keep service you've already paid for. If you're under a device installment plan, the remaining balance becomes due. If something specific prompted this, tell the team — there may be a plan or fix that works better."},
    {"intent": "account",
     "question": "How do I transfer my number to another carrier?",
     "variants": ["I need my account number and PIN for porting out."],
     "answer": "To port your number out, give your new carrier two things: your account number (in the portal under Profile) and your transfer PIN (generate it under Account Security > Transfer PIN). Important: do not cancel your service first — the port itself closes the line automatically, usually within 1 business day. Cancelling early can make the number unrecoverable."},
    {"intent": "account",
     "question": "How do I bring my number from my old carrier?",
     "variants": ["Can I keep my current number if I switch to you?"],
     "answer": "Yes, in most cases you can keep your number. During signup choose 'Bring my own number' and provide your current carrier's account number and transfer PIN (get these from them before starting). Keep your old service active until the port completes — typically within 1 business day. You'll get an SMS confirmation when your number is live with us."},
    {"intent": "account",
     "question": "How do I change my account PIN?",
     "variants": ["I want to update my security PIN."],
     "answer": "Go to Account Security in the portal and choose Change PIN — you'll verify with a code sent to your phone first. Your PIN is required for porting, account changes, and store visits, so choose something memorable but not guessable (avoid birthdays). If you suspect someone knows your PIN, change it immediately."},
    {"intent": "account",
     "question": "Can someone else manage my account?",
     "variants": ["How do I add my spouse as an authorized user?"],
     "answer": "Yes — add an authorized user in the portal under Profile > Authorized Users. They'll be able to make plan changes, upgrade devices, and contact support on the account, but only the primary account holder can cancel service or add lines. You can remove authorized users at any time."},

    # ── General / edge cases ─────────────────────────────────────────────────
    {"intent": "general",
     "question": "I've been a customer for years and this is the worst service ever. Fix it or I'm leaving!",
     "variants": ["This is ridiculous, nothing works and nobody helps me."],
     "answer": "I'm really sorry — that's not the experience we want for you, especially as a long-time customer. Let's fix this properly: tell me the specific issue you're hitting (billing, coverage, device?) and I'll get it to the right team with priority. If you'd like, I can also flag your account for a callback from our customer care team so a person owns this end to end."},
    {"intent": "general",
     "question": "My bill is high AND my data stopped working since I traveled.",
     "variants": ["Two problems: overcharged this month and no internet abroad."],
     "answer": "Those two are likely connected — international roaming both adds charges and behaves differently than home data. First, check your bill's charge breakdown for roaming line items from the travel dates. Second, for data abroad you need roaming enabled in your phone settings and an international plan ($25/month includes data roaming) — without one, data may be blocked to protect you from high charges. I'd suggest adding the plan now, and if any charges look wrong you have 60 days to dispute them."},
    {"intent": "general",
     "question": "What's the weather like tomorrow?",
     "variants": ["Can you recommend a good restaurant nearby?"],
     "answer": "I'm the telecom support assistant, so weather and restaurants are outside what I can help with — but I'm happy to help with anything about your mobile service: billing, coverage, devices, plans, or your account. What can I look into for you?"},
    {"intent": "general",
     "question": "Are you a real person?",
     "variants": ["Am I talking to a bot?"],
     "answer": "I'm a virtual support assistant. I can resolve most billing, network, device, plan, and account questions instantly. If your issue needs a human — like complex disputes or retention offers — I'll connect you to our care team without making you repeat yourself."},
    {"intent": "general",
     "question": "How do I contact a human agent?",
     "variants": ["I want to talk to a real person."],
     "answer": "You can reach our care team by phone or live chat — both are listed under Contact Us in the app and portal, available 7am-11pm daily. If you tell me your issue first, I can route you to the right department directly so you skip the general queue."},
]


def _repo_root_on_path():
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def _kb_chunks_for_intent(intent: str) -> list[str]:
    """KB chunks matching the intent — mirrors what the RAG agent retrieves at
    inference time, so training prompts have the same shape as production ones."""
    _repo_root_on_path()
    from agents.rag_agent import TELECOM_KB
    matching = [k["text"] for k in TELECOM_KB if k["category"] == intent]
    if not matching:  # "general" — RAG still returns top-3 of something
        matching = [TELECOM_KB[0]["text"], TELECOM_KB[3]["text"], TELECOM_KB[10]["text"]]
    return matching[:3]


def get_dataset():
    """Return dataset as instruction-response pairs, expanding question variants.

    The instruction format must match agents/response_generator.py exactly —
    a train/inference prompt mismatch makes the model ignore the RAG context
    and hallucinate policies (observed on Qwen3-14B compare runs). For the
    same reason each question is labeled by the REAL intent classifier, so
    the Intent line at train time always equals what production produces.

    Includes the proprietary internal-knowledge layer (data/internal_kb.py):
    billing codes, CPE hardware, error codes — the fine-tuning testbed.
    """
    _repo_root_on_path()
    from agents.intent_classifier import IntentClassifierAgent
    from data.internal_kb import INTERNAL_QA

    classifier = IntentClassifierAgent()
    samples = []
    for item in TELECOM_QA + INTERNAL_QA:
        questions = [item["question"]] + item.get("variants", [])
        for q in questions:
            intent = classifier.classify(q).intent
            context = "\n".join(f"- {c}" for c in _kb_chunks_for_intent(intent))
            samples.append({
                "instruction": (
                    f"You are a helpful telecom customer support agent. Intent: {intent}.\n\n"
                    f"Relevant knowledge:\n{context}\n\n"
                    f"Customer query: {q}"
                ),
                "output": item["answer"],
                "intent": intent,
            })
    return samples


def feedback_samples():
    """Approved user-feedback pairs (data flywheel) in training format —
    same instruction shape, intent labeled by the real classifier."""
    _repo_root_on_path()
    from agents.intent_classifier import IntentClassifierAgent
    from data.feedback_store import load_approved

    approved = load_approved()
    if not approved:
        return []

    classifier = IntentClassifierAgent()
    samples = []
    for row in approved:
        intent = classifier.classify(row["question"]).intent
        context = "\n".join(f"- {c}" for c in _kb_chunks_for_intent(intent))
        samples.append({
            "instruction": (
                f"You are a helpful telecom customer support agent. Intent: {intent}.\n\n"
                f"Relevant knowledge:\n{context}\n\n"
                f"Customer query: {row['question']}"
            ),
            "output": row["answer"],
            "intent": intent,
        })
    return samples


def get_alpaca_format():
    """Return dataset in Alpaca format for training."""
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


if __name__ == "__main__":
    ds = get_dataset()
    from collections import Counter
    from data.internal_kb import INTERNAL_QA
    print(f"Total training samples: {len(ds)}")
    print(f"Unique answers: {len(TELECOM_QA) + len(INTERNAL_QA)} ({len(INTERNAL_QA)} proprietary)")
    for intent, n in Counter(s["intent"] for s in ds).most_common():
        print(f"  {intent:8s}: {n}")
