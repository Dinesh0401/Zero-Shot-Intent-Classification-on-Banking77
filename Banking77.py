import os
from openai import OpenAI
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = r"C:\Users\sjdin\Task3\banking77_sample_500.csv"
MODELS = ["gpt-5.2", "gpt-5-mini"]

# -----------------------------
# LABEL SET (77)
# -----------------------------
LABELS = [
    "activate_my_card","age_limit","apple_pay_or_google_pay","atm_support",
    "automatic_top_up","balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit","beneficiary_not_allowed",
    "cancel_transfer","card_about_to_expire","card_arrival","card_delivery_estimate",
    "card_linking","card_not_working","card_payment_fee_charged",
    "card_payment_wrong_exchange_rate","card_swallowed","cash_withdrawal_charge",
    "cash_withdrawal_not_recognised","chargeback",
    "charged_card_purchase_wrong_exchange_rate","charged_exchange_rate_wrong",
    "charged_fee","charged_fee_wrong","close_account","compromised_card",
    "contactless_not_working","country_support","declined_card_payment",
    "declined_cash_withdrawal","declined_transfer",
    "direct_debit_payment_not_recognised","disposable_card_limits",
    "edit_personal_details","exchange_charge","exchange_rate","exchange_via_app",
    "extra_charge_on_statement","failed_transfer","fiat_currency_support",
    "get_disposable_virtual_card","get_physical_card","getting_spare_card",
    "getting_virtual_card","lost_or_stolen_card","lost_or_stolen_phone",
    "order_physical_card","passcode_forgotten","pending_card_payment",
    "pending_cash_withdrawal","pending_transfer","pin_blocked","receiving_money",
    "refund_not_received","request_chargeback","reverted_card_payment",
    "supported_cards_and_currencies","terminate_account",
    "top_up_by_bank_transfer_charge","top_up_by_card_charge",
    "top_up_by_cash_or_cheque","top_up_failed","top_up_limits","top_up_reverted",
    "topping_up_by_bank_transfer","unable_to_verify_identity",
    "verify_my_identity","visa_or_mastercard","why_verify_identity",
    "wrong_amount_of_cash_received","wrong_exchange_rate_for_cash_withdrawal"
]

labels_text = ",\n".join(LABELS)

# -----------------------------
# SYSTEM PROMPT (TL-CORRECT)
# -----------------------------
SYSTEM_PROMPT = f"""
You are a banking intent classification system.

Classify the given user query into EXACTLY ONE label
from the predefined list below.

Rules:
- Choose ONLY one label
- Use ONLY labels from the list
- Do NOT explain
- Do NOT invent labels
- If ambiguous, choose the most specific intent

Intent labels:
{labels_text}

Return ONLY the label name.
"""

# -----------------------------
# OPENAI CLIENT
# -----------------------------
# Read API key from environment to avoid committing secrets to source.
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.\nSet it before running the script.")

client = OpenAI(api_key=api_key)

def classify(model_name, query):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# LOAD DATA and EVALUATION
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    # IMPORTANT: category is the true label column
    return list(zip(df["text"], df["category"]))

def evaluate():
    data = load_data()
    results = {model: 0 for model in MODELS}

    for text, true_label in data:
        for model in MODELS:
            pred = classify(model, text)
            if pred == true_label:
                results[model] += 1

    total = len(data)
    print(f"\nTotal samples evaluated: {total}\n")
    for model in MODELS:
        accuracy = (results[model] / total) * 100
        print(f"{model} accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate()
