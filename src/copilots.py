"""AI Copilots for Bank Analysts and Consumers.

Implements Sections 12 and 13 from the project specification:
1. Bank Analyst Copilot:
   - Evidence-grounded case summary.
   - Generates structured, editable Suspicious Activity Report (SAR) narrative.
   - Refers strictly to prior transaction IDs, graph patterns, feature values, and model version.
   - Analyst retains final authority; prototype does not auto-submit.

2. Consumer Safety Copilot:
   - Verifies simulated UPI ID before user confirms payment.
   - Multilingual explanations (English, Hindi, Telugu, Tamil, Marathi, Bengali).
   - If history is sparse or unavailable, explicitly flags insufficient history rather than false reassurance.
   - Clearly disclaims synthetic model evaluation.
"""

from datetime import datetime, timezone
import json
from typing import Dict, Any, List


# Multilingual template dictionary for Consumer Safety Copilot
CONSUMER_SAFETY_TEMPLATES = {
    "en": {
        "title": "UPI Payment Safety Advisory",
        "safe_title": "Verified Counterparty",
        "safe_msg": "This UPI ID has established consistent, normal payment activity with low anomaly indicators.",
        "warning_title": "Suspicious Activity Warning",
        "warning_msg": "Caution: This UPI ID has recently received anomalous high-velocity inflows or exhibits rapid pass-through behavior typical of mule accounts. Please verify the recipient before proceeding.",
        "sparse_title": "Insufficient Transaction History",
        "sparse_msg": "Caution: This UPI ID has very little or no prior transaction history recorded in the network. Exercise caution and verify identity before sending funds to newly created handles.",
        "risk_label": "Synthetic Risk Score",
        "disclaimer": "Notice: This advisory is generated from synthetic model heuristics for demonstration purposes. It does not provide an absolute guarantee of payment safety."
    },
    "hi": {
        "title": "यूपीआई भुगतान सुरक्षा सलाह",
        "safe_title": "सत्यापित खाता",
        "safe_msg": "इस यूपीआई आईडी पर सामान्य लेन-देन का इतिहास देखा गया है और जोखिम का स्तर कम है।",
        "warning_title": "संदिग्ध गतिविधि चेतावनी",
        "warning_msg": "सावधान: इस यूपीआई आईडी में हाल ही में असामान्य रूप से तेज़ लेन-देन या म्यूल खाते जैसी संदिग्ध गतिविधियां पाई गई हैं। कृपया पैसे भेजने से पहले प्राप्तकर्ता की पुष्टि करें।",
        "sparse_title": "अपर्याप्त लेन-देन इतिहास",
        "sparse_msg": "सावधान: इस यूपीआई आईडी का नेटवर्क में बहुत कम या शून्य लेन-देन इतिहास है। नए खातों में भुगतान करते समय सतर्क रहें और पहचान की पुष्टि करें।",
        "risk_label": "सिंथेटिक जोखिम स्कोर",
        "disclaimer": "सूचना: यह सलाह केवल प्रदर्शन के लिए मॉडल पर आधारित है। यह वास्तविक भुगतान सुरक्षा की पूर्ण गारंटी नहीं देता है।"
    },
    "te": {
        "title": "యూపీఐ చెల్లింపు భద్రతా హెచ్చరిక",
        "safe_title": "ధృవీకరించబడిన ఖాతా",
        "safe_msg": "ఈ యూపీఐ ఐడీ సాధారణ లావాదేవీల చరిత్రను కలిగి ఉంది మరియు ముప్పు తక్కువగా ఉంది.",
        "warning_title": "అనుమానాస్పద లావాదేవీ హెచ్చరిక",
        "warning_msg": "హెచ్చరిక: ఈ యూపీఐ ఖాతాలోకి ఇటీవల అనుమానాస్పదంగా భారీ నిధుల ప్రవాహం లేదా మ్యూల్ ఖాతా లక్షణాలు గుర్తించబడ్డాయి. దయచేసి డబ్బులు పంపే ముందు గ్రహీతను నిర్ధారించుకోండి.",
        "sparse_title": "పరిమిత లావాదేవీల చరిత్ర",
        "sparse_msg": "హెచ్చరిక: ఈ యూపీఐ ఐడీకి తగినంత గత లావాదేవీల చరిత్ర లేదు. కొత్త లేదా ధృవీకరించని ఖాతాలకు నగదు బదిలీ చేసేటప్పుడు జాగ్రత్త వహించండి.",
        "risk_label": "రిస్క్ స్కోరు",
        "disclaimer": "గమనిక: ఇది సింథటిక్ మోడల్ సూచన మాత్రమే. ఇది లావాదేవీ యొక్క పూర్తి భద్రతకు హామీ ఇవ్వదు."
    },
    "ta": {
        "title": "யுபிஐ கட்டண பாதுகாப்பு எச்சரிக்கை",
        "safe_title": "சரிபார்க்கப்பட்ட கணக்கு",
        "safe_msg": "இந்த யுபிஐ ஐடி வழக்கமான பரிவர்த்தனை வரலாற்றைக் கொண்டுள்ளது, ஆபத்து குறைவு.",
        "warning_title": "சந்தேகத்திற்கிடமான நடவடிக்கை எச்சரிக்கை",
        "warning_msg": "எச்சரிக்கை: இந்த யுபிஐ ஐடியில் வழக்கத்திற்கு மாறான விரைவான பணப் பரிமாற்றம் காணப்படுகிறது. பணத்தை அனுப்பும் முன் பெறுநரை உறுதிப்படுத்தவும்.",
        "sparse_title": "போதிய பரிவர்த்தனை வரலாறு இல்லை",
        "sparse_msg": "எச்சரிக்கை: இந்த யுபிஐ ஐடிக்கு போதுமான பரிவர்த்தனை வரலாறு இல்லை. புதிய கணக்குகளுக்கு பணம் அனுப்பும்போது கவனமாக இருக்கவும்.",
        "risk_label": "ஆபத்து மதிப்பீடு",
        "disclaimer": "அறிவிப்பு: இது மாதிரி மதிப்பீடு மட்டுமே; முழுமையான உத்தரவாதம் அல்ல."
    },
    "mr": {
        "title": "यूपीआय पेमेंट सुरक्षा सल्ला",
        "safe_title": "सत्यापित खाते",
        "safe_msg": "या यूपीआय आयडीवर सामान्य व्यवहार इतिहास आढळला असून जोखीम कमी आहे.",
        "warning_title": "संशयास्पद व्यवहार इशारा",
        "warning_msg": "सावधान: या यूपीआय आयडीवर संशयास्पद वेगाने पैसे जमा होणे किंवा म्यूल खात्यासारखे वर्तन आढळले आहे. कृपया पैसे पाठवण्यापूर्वी पडताळणी करा.",
        "sparse_title": "अपुरा व्यवहार इतिहास",
        "sparse_msg": "सावधान: या यूपीआय आयडीचा नेटवर्कवर फारच कमी किंवा कोणताही इतिहास नाही. नवीन खात्यांवर पैसे पाठवताना काळजी घ्या.",
        "risk_label": "जोखीम गुण",
        "disclaimer": "सूचना: हा सल्ला केवळ प्रात्यक्षिकासाठी तयार केला आहे."
    },
    "bn": {
        "title": "ইউপিআই পেমেন্ট নিরাপত্তা পরামর্শ",
        "safe_title": "যাচাইকৃত অ্যাকাউন্ট",
        "safe_msg": "এই ইউপিআই আইডিটিতে স্বাভাবিক লেনদেনের ইতিহাস রয়েছে এবং ঝুঁকি কম।",
        "warning_title": "সন্দেহজনক কার্যকলাপ সতর্কতা",
        "warning_msg": "সতর্কতা: এই ইউপিআই আইডিতে অস্বাভাবিক দ্রুত লেনদেন লক্ষ্য করা গেছে। টাকা পাঠানোর আগে প্রাপকের পরিচয় নিশ্চিত করুন।",
        "sparse_title": "অপর্যাপ্ত লেনদেনের ইতিহাস",
        "sparse_msg": "সতর্কতা: এই ইউপিআই আইডির পর্যাপ্ত পূর্ববর্তী লেনদেনের রেকর্ড নেই। নতুন অ্যাকাউন্টে টাকা পাঠানোর সময় সতর্ক থাকুন।",
        "risk_label": "ঝুঁকি স্কোর",
        "disclaimer": "বিজ্ঞপ্তি: এটি শুধুমাত্র নির্দেশনামূলক ঝুঁকি মূল্যায়ন।"
    }
}


class BankAnalystCopilot:
    """Generates structured Suspicious Activity Report (SAR) case packages for compliance investigators."""

    @staticmethod
    def generate_sar_summary(case_data: Dict[str, Any]) -> Dict[str, Any]:
        tx_id = case_data.get("transaction_id", "N/A")
        sender = case_data.get("sender_upi", "unknown@upi")
        receiver = case_data.get("receiver_upi", "unknown@upi")
        amount = case_data.get("amount", 0.0)
        risk_score = case_data.get("risk_score", 0.0)
        risk_level = case_data.get("risk_level", "UNKNOWN")
        cutoff = case_data.get("graph_cutoff_timestamp", "N/A")
        reasons = case_data.get("main_reasons", [])
        supporting_ids = case_data.get("supporting_transaction_ids", [])
        subgraph = case_data.get("gnn_influential_subgraph", [])
        shap_feats = case_data.get("tree_shap_explanation", {}).get("top_features", [])
        model_version = "GraphSAGE-v2 / XGBoost-AMLSim-v1"

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Draft structured SAR Narrative
        narrative_lines = [
            f"=== SUSPICIOUS ACTIVITY REPORT (SAR) INVESTIGATION BRIEF ===",
            f"Filing Reference: SAR-UPI-2024-{tx_id}",
            f"Generated At: {now_str}",
            f"Scoring Engine: {model_version}",
            f"Graph History Cutoff: {cutoff} UTC",
            "",
            "1. SUBJECT & TRANSACTION DETAILS:",
            f"   - Target Transaction ID: #{tx_id}",
            f"   - Originating VPA: {sender}",
            f"   - Beneficiary VPA: {receiver}",
            f"   - Transfer Value: INR {amount:,.2f}",
            f"   - Assessed Risk Score: {risk_score:.4f} ({risk_level} RISK)",
            "",
            "2. SUMMARY OF SUSPICIOUS ACTIVITY:",
            f"   On or before time cutoff {cutoff}, beneficiary account '{receiver}' was flagged for anomalous",
            "   payment patterns indicative of digital mule account activity. Historical graph traversal reveals:",
        ]

        for r in reasons:
            narrative_lines.append(f"   • {r}")

        narrative_lines += [
            "",
            "3. GRAPH EVIDENCE & SUPPORTING TRANSACTIONS:",
            f"   GNNExplainer identified {len(subgraph)} prior transaction edge(s) as primary topological evidence:"
        ]

        if len(subgraph) > 0:
            for edge in subgraph:
                narrative_lines.append(
                    f"   • Tx #{edge.get('prior_transaction_id')}: {edge.get('source_upi')} -> {edge.get('target_upi')} "
                    f"[INR {edge.get('amount', 0):,.2f}] (Role: {edge.get('pattern_role')}, GNN Weight: {edge.get('gnn_importance_weight')})"
                )
        else:
            narrative_lines.append("   • No prior multi-hop transactions found strictly prior to cutoff; flagged primarily via tabular velocity.")

        narrative_lines += [
            "",
            "4. KEY MODEL PREDICTOR DRIVERS (SHAP ATTRIBUTION):",
        ]
        for f in shap_feats[:4]:
            narrative_lines.append(f"   • {f['feature']}: {f['feature_value']} (Impact: {f['shap_attribution']:+0.3f}, {f['direction']})")

        narrative_lines += [
            "",
            "5. INVESTIGATOR RECOMMENDATION (EDITABLE):",
            "   [X] Place temporary debit restriction on beneficiary VPA pending KYC verification.",
            "   [X] Issue alert to counterparty sponsoring bank regarding potential mule layering.",
            "   [ ] Request source-of-funds documentation from originating account.",
            "   [ ] Escalate for FIU-IND (Financial Intelligence Unit - India) formal regulatory filing.",
            "",
            "INVESTIGATOR SIGN-OFF:",
            "Analyst ID: ____________________    Date: ______________    Decision: [Approved / Dismissed]"
        ]

        sar_text = "\n".join(narrative_lines)

        return {
            "sar_id": f"SAR-UPI-2024-{tx_id}",
            "generated_at": now_str,
            "transaction_id": tx_id,
            "target_account": receiver,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "supporting_transaction_ids": supporting_ids,
            "model_version": model_version,
            "editable_sar_text": sar_text
        }


class ConsumerSafetyCopilot:
    """Provides plain-language, multilingual payment safety advisories for consumer UPI apps."""

    @staticmethod
    def check_recipient_safety(
        recipient_upi: str,
        recent_risk_score: float = 0.05,
        total_prior_events: int = 15,
        language: str = "en"
    ) -> Dict[str, Any]:
        lang = language.lower() if language.lower() in CONSUMER_SAFETY_TEMPLATES else "en"
        tmpl = CONSUMER_SAFETY_TEMPLATES[lang]

        # Determine status
        is_sparse = total_prior_events < 3
        is_high_risk = recent_risk_score >= 0.50

        if is_sparse:
            status = "SPARSE_HISTORY"
            category = "CAUTION"
            title = tmpl["sparse_title"]
            message = tmpl["sparse_msg"]
            badge_color = "amber"
        elif is_high_risk:
            status = "HIGH_RISK"
            category = "WARNING"
            title = tmpl["warning_title"]
            message = tmpl["warning_msg"]
            badge_color = "red"
        else:
            status = "NORMAL"
            category = "SAFE"
            title = tmpl["safe_title"]
            message = tmpl["safe_msg"]
            badge_color = "green"

        return {
            "recipient_upi": recipient_upi,
            "language": lang,
            "status": status,
            "safety_category": category,
            "badge_color": badge_color,
            "title": title,
            "advisory_message": message,
            "synthetic_risk_score": round(recent_risk_score, 4),
            "recorded_transactions_count": total_prior_events,
            "is_history_sparse": is_sparse,
            "disclaimer": tmpl["disclaimer"]
        }


def test_copilots():
    print("Testing Bank Analyst Copilot...")
    mock_case = {
        "transaction_id": 179904,
        "sender_upi": "acc_17800@paytm",
        "receiver_upi": "acc_08676@okhdfcbank",
        "amount": 14.03,
        "risk_score": 0.9832,
        "risk_level": "HIGH",
        "graph_cutoff_timestamp": 1725804211,
        "main_reasons": [
            "Rapid fan-in: Beneficiary received multiple inbound credits within 2 hours",
            "Immediate layering: Funds routed downstream to counterparty accounts"
        ],
        "supporting_transaction_ids": [154566, 171490, 173795],
        "gnn_influential_subgraph": [
            {"prior_transaction_id": 154566, "source_upi": "acc_08676@okhdfcbank", "target_upi": "acc_03142@paytm", "amount": 256.70, "pattern_role": "Layering Outflow", "gnn_importance_weight": 0.92}
        ],
        "tree_shap_explanation": {
            "top_features": [
                {"feature": "log1p_amount", "feature_value": 2.71, "shap_attribution": 6.35, "direction": "Pushes Risk HIGHER"}
            ]
        }
    }
    sar = BankAnalystCopilot.generate_sar_summary(mock_case)
    print(f"SAR Report Reference: {sar['sar_id']}")
    print(sar["editable_sar_text"][:350] + "...\n")

    print("Testing Consumer Safety Copilot in multiple languages...")
    for l in ["en", "hi", "te"]:
        adv = ConsumerSafetyCopilot.check_recipient_safety("new_vendor@upi", recent_risk_score=0.1, total_prior_events=1, language=l)
        print(f"[{l.upper()}] {adv['title']}: {adv['advisory_message']}")


if __name__ == "__main__":
    test_copilots()

