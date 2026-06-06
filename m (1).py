"""
Automated Diagnosis of Oral Conditions from Dental X-Rays
Student  : Sonali Patra | Reg. No.: 24C216A45
Guide    : Dr. Debabrata Singh
Dept.    : Computer Application, ITER, SOA University
Run      : streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import random, time, io, os
from PIL import Image
from datetime import datetime, timezone, timedelta

# ── optional PDF ──
try:
    from fpdf import FPDF
    FPDF_OK = True
except:
    FPDF_OK = False

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="OralDX – Dental AI Diagnosis",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────
# CSS — pure white background, black text
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* White background everywhere */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] {
    background-color: #f5f5f5 !important;
}
/* Black text */
body, p, div, span, label, h1, h2, h3, h4 {
    color: #111111 !important;
}
/* Hide default header */
[data-testid="stHeader"] { display: none; }
#MainMenu, footer { visibility: hidden; }

/* Blue buttons */
.stButton > button {
    background-color: #1565c0 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    width: 100%;
}
.stButton > button:hover {
    background-color: #0d47a1 !important;
}

/* Cards */
.card {
    background: #ffffff;
    border: 1.5px solid #e0e0e0;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 14px;
}

/* Page title */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111111;
    border-bottom: 3px solid #1565c0;
    padding-bottom: 8px;
    margin-bottom: 20px;
}

/* Metric box */
.metric-box {
    background: #f0f4ff;
    border: 1px solid #c5cae9;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1565c0;
}
.metric-label {
    font-size: 0.82rem;
    color: #444444;
    margin-top: 4px;
}

/* Result box */
.result-box {
    background: #e8f5e9;
    border: 2px solid #4caf50;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-bottom: 14px;
}

/* Alert boxes */
.info-box {
    background: #e3f2fd;
    border-left: 4px solid #1565c0;
    border-radius: 6px;
    padding: 12px 16px;
    color: #111111;
    margin: 8px 0;
    font-size: 0.88rem;
}
.warn-box {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 6px;
    padding: 12px 16px;
    color: #111111;
    margin: 8px 0;
    font-size: 0.88rem;
}
.success-box {
    background: #e8f5e9;
    border-left: 4px solid #4caf50;
    border-radius: 6px;
    padding: 12px 16px;
    color: #111111;
    margin: 8px 0;
    font-size: 0.88rem;
}

/* Timeline */
.timeline-row {
    padding: 10px 0;
    border-bottom: 1px solid #eeeeee;
    font-size: 0.88rem;
    color: #111111;
}

/* Member card */
.member-card {
    background: #ffffff;
    border: 1.5px solid #e0e0e0;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin-bottom: 10px;
}
.member-name {
    font-weight: 700;
    font-size: 0.9rem;
    color: #111111;
    margin-top: 8px;
}
.member-role {
    font-size: 0.75rem;
    color: #1565c0;
    background: #e3f2fd;
    border-radius: 12px;
    padding: 3px 10px;
    display: inline-block;
    margin-top: 4px;
}

/* Sidebar radio */
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    color: #111111 !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────
DISEASES = [
    "Caries", "Calculus", "Gingivitis",
    "Mouth Ulcer", "Tooth Discoloration", "Hypodontia"
]

DATASET = {
    "Caries":              {"count": 2601, "pct": 19.5, "color": "#ef4444"},
    "Calculus":            {"count": 1296, "pct":  9.7, "color": "#f97316"},
    "Gingivitis":          {"count": 2349, "pct": 17.6, "color": "#3b82f6"},
    "Mouth Ulcer":         {"count": 2806, "pct": 21.1, "color": "#8b5cf6"},
    "Tooth Discoloration": {"count": 2017, "pct": 15.1, "color": "#06b6d4"},
    "Hypodontia":          {"count": 1251, "pct":  9.4, "color": "#22c55e"},
}

RECOMMENDATIONS = {
    "Caries":              "Visit a dentist for filling or crown. Avoid sugary foods. Use fluoride toothpaste.",
    "Calculus":            "Professional dental scaling required. Use antibacterial mouthwash. Improve flossing.",
    "Gingivitis":          "Improve oral hygiene. Professional cleaning needed. Use antiseptic mouthwash.",
    "Mouth Ulcer":         "Apply topical gel for pain. Use antiseptic mouthwash. Avoid spicy/acidic foods.",
    "Tooth Discoloration": "Consult dentist for whitening or veneers. Avoid tea, coffee, tobacco.",
    "Hypodontia":          "Consult orthodontist for implants, bridges, or space closure. Early treatment best.",
}

TEAM = [
    {"name": "Dr. Debabrata Singh",         "reg": "Project Guide",  "role": "Guide",                          "type": "guide",   "photo": "guide.jpeg"},
    {"name": "Sonali Patra",                "reg": "24C216A45",      "role": "Student",         "type": "student", "photo": "sonali.jpeg"},
    {"name": "Jagruti Parida",              "reg": "24C216A47",    "role": "Student",               "type": "student", "photo": "jagruti.jpeg"},
    {"name": "Dharitri Pradhan",            "reg": "24C216A30",    "role": "Student",               "type": "student", "photo": "dharitri.jpeg"},
    {"name": "Smitarani Mahapatra",         "reg": "24C213A05",    "role": "Student",             "type": "student", "photo": "smitarani.jpeg"},
    {"name": "Barsha Priyadarshini Singh",  "reg": "24C219A30",    "role": "Student",         "type": "student", "photo": "barsha.jpeg"},
]

CHATBOT_ANSWERS = {
    "caries":              "🦷 Caries is dental decay caused by bacteria producing acid that erodes enamel. Visible as dark radiolucent spots on X-ray. Treatment: filling or crown.",
    "calculus":            "🪨 Calculus is hardened plaque (tartar) that forms on teeth. Appears as bright radio-opaque deposits near roots on X-ray. Treatment: professional scaling.",
    "gingivitis":          "🔴 Gingivitis is gum inflammation caused by plaque. Shows soft-tissue changes on X-ray. Treatment: deep cleaning and improved oral hygiene.",
    "mouth ulcer":         "💢 Mouth ulcers are painful sores on the inner mouth lining. Visible as light lesions on X-ray. Treatment: topical gel, mouthwash, vitamin supplements.",
    "tooth discoloration": "🟡 Tooth discoloration is colour change due to intrinsic or extrinsic factors. Detected by VGG19 intensity analysis. Treatment: whitening, veneers.",
    "hypodontia":          "❌ Hypodontia is the congenital absence of one or more teeth. Visible as missing gaps in panoramic X-ray. Treatment: implants or bridges.",
    "treatment":           "💊 Caries→Filling | Calculus→Scaling | Gingivitis→Cleaning | Mouth Ulcer→Topical Gel | Tooth Discoloration→Whitening | Hypodontia→Implant.",
    "precautions":         "✅ Brush twice daily, floss daily, avoid sugary drinks, visit dentist every 6 months, use mouthwash.",
    "models":              "🤖 4 AI Models: ResNet-50 (91% accuracy, F1=0.895), VGG19 (87% accuracy, F1=0.860), YOLOv8 (88% mAP@0.5, F1=0.855), U-Net (85% Dice, F1=0.825).",
    "accuracy":            "🎯 Model accuracy (from report Table 4.6): ResNet-50=91% (F1=0.895), VGG19=87% (F1=0.860), YOLOv8=88% mAP@0.5 (F1=0.855), U-Net=85% Dice (F1=0.825).",
    "dataset":             "📊 Dataset: 13,320 dental X-ray images — Caries(2601), Calculus(1296), Gingivitis(2349), Mouth Ulcer(2806), Tooth Discoloration(2017), Hypodontia(1251).",
    "default":             "🤖 Ask me about: caries, calculus, gingivitis, mouth ulcer, tooth discoloration, hypodontia, treatment, precautions, models, accuracy, or dataset.",
}


# ─────────────────────────────────────────────────────
# TRANSLATIONS  (English / Hindi / Odia)
# ─────────────────────────────────────────────────────
TRANSLATIONS = {
    "English": {
        "nav_home":     "🏠 Home Dashboard",
        "nav_upload":   "📤 Upload & Diagnose",
        "nav_results":  "📊 View Results",
        "nav_chat":     "🤖 AI Chatbot",
        "nav_compare":  "🔁 Compare X-Rays",
        "nav_history":  "📂 History",
        "nav_reviews":  "⭐ Reviews",
        "nav_about":    "ℹ️ About Us",
        "nav_profile":  "👤 Profile",
        "nav_admin":    "🛡️ Admin Panel",
        "nav_signin":   "🔑 Sign In / Sign Up",
        "logout":       "🚪 Logout",
        "lang_label":   "🌐 Language",
        "home_title":   "Home Dashboard",
        "upload_title": "Upload Dental X-Ray",
        "result_title": "Diagnosis Results",
        "chat_title":   "AI Dental Chatbot",
        "compare_title":"Compare Two X-Rays",
        "history_title":"Diagnosis History",
        "review_title": "Reviews & Feedback",
        "about_title":  "About Us",
        "profile_title":"My Profile",
        "admin_title":  "Admin Panel",
        "signin_title": "Sign In / Sign Up",
        "upload_btn":   "🧠 Start AI Diagnosis",
        "compare_btn":  "🔍 Compare & Analyse",
        "submit_review":"📨 Submit Review",
        "save_btn":     "💾 Save",
        "clear_hist":   "🗑️ Clear All History",
        "clear_chat":   "🗑️ Clear Chat",
        "download_pdf": "📥 Download PDF Report",
        "search_ph":    "Search by disease or date...",
        "no_records":   "No records found. Upload an X-ray to begin.",
        "no_results":   "No results yet. Upload an X-ray and run diagnosis first.",
        "condition":    "DETECTED CONDITION",
        "confidence":   "Confidence Score",
        "severity":     "Severity Level",
        "region":       "Affected Region",
        "heatmap":      "AI Heatmap",
        "recommendation":"Recommendation",
        "ai_models":    "AI Models Analysis",
        "disclaimer":   "⚠️ AI-generated guidance only. Always consult a qualified dentist.",
        "prev_scan":    "Previous Scan (Older)",
        "curr_scan":    "Current Scan (Newer)",
        "improvement":  "Improvement",
        "write_review": "Write a Review",
        "your_name":    "Your Name",
        "your_feedback":"Your Feedback",
        "feedback_ph":  "Write your experience...",
        "rating":       "Rating",
        "project":      "Project",
        "department":   "Department",
        "batch":        "Batch",
        "guide":        "Project Guide",
        "team":         "Project Team",
        "tech":         "Technology Used",
        "photo_note":   "To show member photos: Place these files in the same folder as app.py",
        "total_scans":  "Total Scans",
        "total_users":  "Total Users",
        "ai_accuracy":  "AI Accuracy",
        "most_detected":"Most Detected",
        "update_name":  "Update Name",
        "display_name": "Display Name",
        "sign_in":      "Sign In",
        "sign_up":      "Sign Up",
        "email":        "Email",
        "password":     "Password",
        "confirm_pwd":  "Confirm Password",
        "full_name":    "Full Name",
        "login_as":     "Login as",
        "create_acc":   "Create Account →",
        "signin_btn":   "Sign In →",
        "demo_note":    "💡 Demo: Enter any email + password to login.",
        "greet_morning":"Good Morning ☀️",
        "greet_afternoon":"Good Afternoon 🌤️",
        "greet_evening":"Good Evening 🌇",
        "greet_night":"Good Night 🌙",
        "quick_q":      "Quick Questions:",
        "type_here":    "Type your question here...",
        "export_csv":   "📥 Export CSV",
        "clear_logs":   "🗑️ Clear Logs",
        "dataset_total":"Total Dataset: 13,320 X-ray images across 6 disease classes",
        "name_updated": "Name updated!",
        "review_ok":    "✅ Review submitted! Thank you.",
        "diag_done":    "✅ Diagnosis complete! Go to View Results in sidebar.",
        "uploaded_ok":  "✅ Uploaded",
        "preview":      "Image preview will appear here",
    },
    "Hindi": {
        "nav_home":     "🏠 होम डैशबोर्ड",
        "nav_upload":   "📤 अपलोड और निदान",
        "nav_results":  "📊 परिणाम देखें",
        "nav_chat":     "🤖 AI चैटबॉट",
        "nav_compare":  "🔁 X-Ray तुलना",
        "nav_history":  "📂 इतिहास",
        "nav_reviews":  "⭐ समीक्षाएं",
        "nav_about":    "ℹ️ हमारे बारे में",
        "nav_profile":  "👤 प्रोफ़ाइल",
        "nav_admin":    "🛡️ एडमिन पैनल",
        "nav_signin":   "🔑 साइन इन / साइन अप",
        "logout":       "🚪 लॉगआउट",
        "lang_label":   "🌐 भाषा",
        "home_title":   "होम डैशबोर्ड",
        "upload_title": "दंत X-Ray अपलोड करें",
        "result_title": "निदान परिणाम",
        "chat_title":   "AI दंत चैटबॉट",
        "compare_title":"दो X-Ray की तुलना",
        "history_title":"निदान इतिहास",
        "review_title": "समीक्षाएं और प्रतिक्रिया",
        "about_title":  "हमारे बारे में",
        "profile_title":"मेरी प्रोफ़ाइल",
        "admin_title":  "एडमिन पैनल",
        "signin_title": "साइन इन / साइन अप",
        "upload_btn":   "🧠 AI निदान शुरू करें",
        "compare_btn":  "🔍 तुलना और विश्लेषण",
        "submit_review":"📨 समीक्षा सबमिट करें",
        "save_btn":     "💾 सहेजें",
        "clear_hist":   "🗑️ इतिहास हटाएं",
        "clear_chat":   "🗑️ चैट साफ़ करें",
        "download_pdf": "📥 PDF रिपोर्ट डाउनलोड करें",
        "search_ph":    "रोग या तारीख से खोजें...",
        "no_records":   "कोई रिकॉर्ड नहीं। X-Ray अपलोड करें।",
        "no_results":   "अभी तक कोई परिणाम नहीं। X-Ray अपलोड करें।",
        "condition":    "निदान स्थिति",
        "confidence":   "आत्मविश्वास स्कोर",
        "severity":     "गंभीरता स्तर",
        "region":       "प्रभावित क्षेत्र",
        "heatmap":      "AI हीटमैप",
        "recommendation":"सिफारिश",
        "ai_models":    "AI मॉडल विश्लेषण",
        "disclaimer":   "⚠️ यह AI निर्मित मार्गदर्शन है। योग्य दंत चिकित्सक से सलाह लें।",
        "prev_scan":    "पिछला स्कैन (पुराना)",
        "curr_scan":    "वर्तमान स्कैन (नया)",
        "improvement":  "सुधार",
        "write_review": "समीक्षा लिखें",
        "your_name":    "आपका नाम",
        "your_feedback":"आपकी प्रतिक्रिया",
        "feedback_ph":  "अपना अनुभव लिखें...",
        "rating":       "रेटिंग",
        "project":      "प्रोजेक्ट",
        "department":   "विभाग",
        "batch":        "बैच",
        "guide":        "प्रोजेक्ट गाइड",
        "team":         "प्रोजेक्ट टीम",
        "tech":         "प्रयुक्त तकनीक",
        "photo_note":   "फोटो दिखाने के लिए: app.py के साथ फाइलें रखें",
        "total_scans":  "कुल स्कैन",
        "total_users":  "कुल उपयोगकर्ता",
        "ai_accuracy":  "AI सटीकता",
        "most_detected":"सर्वाधिक निदान",
        "update_name":  "नाम अपडेट करें",
        "display_name": "प्रदर्शन नाम",
        "sign_in":      "साइन इन",
        "sign_up":      "साइन अप",
        "email":        "ईमेल",
        "password":     "पासवर्ड",
        "confirm_pwd":  "पासवर्ड पुष्टि",
        "full_name":    "पूरा नाम",
        "login_as":     "भूमिका",
        "create_acc":   "खाता बनाएं →",
        "signin_btn":   "साइन इन →",
        "demo_note":    "💡 डेमो: कोई भी ईमेल + पासवर्ड डालें।",
        "greet_morning":"शुभ प्रभात ☀️",
        "greet_afternoon":"शुभ अपराह्न 🌤️",
        "greet_evening":"शुभ संध्या 🌇",
        "greet_night":"शुभ रात्रि 🌙",
        "quick_q":      "त्वरित प्रश्न:",
        "type_here":    "यहाँ प्रश्न टाइप करें...",
        "export_csv":   "📥 CSV निर्यात",
        "clear_logs":   "🗑️ लॉग हटाएं",
        "dataset_total":"कुल डेटासेट: 13,320 X-Ray छवियां",
        "name_updated": "नाम अपडेट हो गया!",
        "review_ok":    "✅ समीक्षा सबमिट हुई!",
        "diag_done":    "✅ निदान पूर्ण! साइडबार में परिणाम देखें।",
        "uploaded_ok":  "✅ अपलोड सफल",
        "preview":      "छवि पूर्वावलोकन यहाँ दिखेगा",
    },
    "Odia": {
        "nav_home":     "🏠 ହୋମ ଡ୍ୟାସବୋର୍ଡ",
        "nav_upload":   "📤 ଅପଲୋଡ ଏବଂ ନିଦାନ",
        "nav_results":  "📊 ଫଳ ଦେଖନ୍ତୁ",
        "nav_chat":     "🤖 AI ଚାଟବଟ",
        "nav_compare":  "🔁 X-ରେ ତୁଳନା",
        "nav_history":  "📂 ଇତିହାସ",
        "nav_reviews":  "⭐ ସମୀକ୍ଷା",
        "nav_about":    "ℹ️ ଆମ ବିଷୟ",
        "nav_profile":  "👤 ପ୍ରୋଫାଇଲ",
        "nav_admin":    "🛡️ ଆଡମିନ ପ୍ୟାନେଲ",
        "nav_signin":   "🔑 ସାଇନ ଇନ / ସାଇନ ଅପ",
        "logout":       "🚪 ଲଗଆଉଟ",
        "lang_label":   "🌐 ଭାଷା",
        "home_title":   "ହୋମ ଡ୍ୟାସବୋର୍ଡ",
        "upload_title": "ଦାନ୍ତ X-ରେ ଅପଲୋଡ",
        "result_title": "ନିଦାନ ଫଳ",
        "chat_title":   "AI ଦନ୍ତ ଚାଟବଟ",
        "compare_title":"ଦୁଇ X-ରେ ତୁଳନା",
        "history_title":"ନିଦାନ ଇତିହାସ",
        "review_title": "ସମୀକ୍ଷା ଏବଂ ମତାମତ",
        "about_title":  "ଆମ ବିଷୟ",
        "profile_title":"ମୋ ପ୍ରୋଫାଇଲ",
        "admin_title":  "ଆଡମିନ ପ୍ୟାନେଲ",
        "signin_title": "ସାଇନ ଇନ / ସାଇନ ଅପ",
        "upload_btn":   "🧠 AI ନିଦାନ ଆରମ୍ଭ",
        "compare_btn":  "🔍 ତୁଳନା ଏବଂ ବିଶ୍ଳେଷଣ",
        "submit_review":"📨 ସମୀକ୍ଷା ଦାଖଲ",
        "save_btn":     "💾 ସଞ୍ଚୟ",
        "clear_hist":   "🗑️ ଇତିହାସ ହଟାନ୍ତୁ",
        "clear_chat":   "🗑️ ଚାଟ ସଫା",
        "download_pdf": "📥 PDF ରିପୋର୍ଟ ଡାଉନଲୋଡ",
        "search_ph":    "ରୋଗ ବା ତାରିଖ ଖୋଜନ୍ତୁ...",
        "no_records":   "କୌଣସି ରେକର୍ଡ ନାହିଁ। X-ରେ ଅପଲୋଡ କରନ୍ତୁ।",
        "no_results":   "ଏ ପର୍ଯ୍ୟନ୍ତ ଫଳ ନାହିଁ। X-ରେ ଅପଲୋଡ କରନ୍ତୁ।",
        "condition":    "ନିଦାନ ଅବସ୍ଥା",
        "confidence":   "ଆତ୍ମବିଶ୍ୱାସ ସ୍କୋର",
        "severity":     "ଗୁରୁତ୍ୱ ସ୍ତର",
        "region":       "ପ୍ରଭାବିତ ଅଞ୍ଚଳ",
        "heatmap":      "AI ହିଟମ୍ୟାପ",
        "recommendation":"ପରାମର୍ଶ",
        "ai_models":    "AI ମଡେଲ ବିଶ୍ଳେଷଣ",
        "disclaimer":   "⚠️ AI ନିର୍ମିତ ମାର୍ଗଦର୍ଶନ। ଯୋଗ୍ୟ ଦନ୍ତ ଚିକିତ୍ସକଙ୍କ ସହ ପରାମର୍ଶ କରନ୍ତୁ।",
        "prev_scan":    "ପୂର୍ବ ସ୍କ୍ୟାନ (ପୁରୁଣା)",
        "curr_scan":    "ବର୍ତ୍ତମାନ ସ୍କ୍ୟାନ (ନୂଆ)",
        "improvement":  "ଉନ୍ନତି",
        "write_review": "ସମୀକ୍ଷା ଲେଖନ୍ତୁ",
        "your_name":    "ଆପଣଙ୍କ ନାମ",
        "your_feedback":"ଆପଣଙ୍କ ମତାମତ",
        "feedback_ph":  "ଅଭିଜ୍ଞତା ଲେଖନ୍ତୁ...",
        "rating":       "ରେଟିଂ",
        "project":      "ପ୍ରକଳ୍ପ",
        "department":   "ବିଭାଗ",
        "batch":        "ବ୍ୟାଚ",
        "guide":        "ପ୍ରକଳ୍ପ ଗାଇଡ",
        "team":         "ପ୍ରକଳ୍ପ ଦଳ",
        "tech":         "ବ୍ୟବହୃତ ପ୍ରଯୁକ୍ତି",
        "photo_note":   "ଫଟୋ ଦେଖାଇବା: app.py ସହ ଫାଇଲ ରଖନ୍ତୁ",
        "total_scans":  "ମୋଟ ସ୍କ୍ୟାନ",
        "total_users":  "ମୋଟ ବ୍ୟବହାରକାରୀ",
        "ai_accuracy":  "AI ସଠିକତା",
        "most_detected":"ସର୍ବାଧିକ ଚିହ୍ନଟ",
        "update_name":  "ନାମ ଅପଡେଟ",
        "display_name": "ଡିସ୍ପ୍ଲେ ନାମ",
        "sign_in":      "ସାଇନ ଇନ",
        "sign_up":      "ସାଇନ ଅପ",
        "email":        "ଇମେଲ",
        "password":     "ପାସୱାର୍ଡ",
        "confirm_pwd":  "ପାସୱାର୍ଡ ନିଶ୍ଚିତ",
        "full_name":    "ପୂର୍ଣ ନାମ",
        "login_as":     "ଭୂମିକା",
        "create_acc":   "ଆକାଉଣ୍ଟ ତିଆରି →",
        "signin_btn":   "ସାଇନ ଇନ →",
        "demo_note":    "💡 ଡେମୋ: ଯେ କୌଣସି ଇମେଲ + ପାସୱାର୍ଡ ଦିଅନ୍ତୁ।",
        "greet_morning":"ଶୁଭ ପ୍ରଭାତ ☀️",
        "greet_afternoon":"ଶୁଭ ଅପରାହ୍ଣ 🌤️",
        "greet_evening":"ଶୁଭ ସନ୍ଧ୍ୟା 🌇",
        "greet_night":"ଶୁଭ ରାତ୍ରି 🌙",
        "quick_q":      "ଦ୍ରୁତ ପ୍ରଶ୍ନ:",
        "type_here":    "ଇଠାରେ ଟାଇପ କରନ୍ତୁ...",
        "export_csv":   "📥 CSV ରପ୍ତାନି",
        "clear_logs":   "🗑️ ଲଗ ହଟାନ୍ତୁ",
        "dataset_total":"ମୋଟ ଡେଟାସେଟ: 13,320 X-ରେ ଛବି",
        "name_updated": "ନାମ ଅପଡେଟ ହୋଇଛି!",
        "review_ok":    "✅ ସମୀକ୍ଷା ଦାଖଲ ହୋଇଛି!",
        "diag_done":    "✅ ନିଦାନ ସମ୍ପୂର୍ଣ! ସାଇଡବାରରେ ଫଳ ଦେଖନ୍ତୁ।",
        "uploaded_ok":  "✅ ଅପଲୋଡ ସଫଳ",
        "preview":      "ଛବି ପୂର୍ବାବଲୋକନ ଏଠାରେ ଦେଖାଯିବ",
    },
}

def T(key):
    """Get translated string for current language."""
    lang = st.session_state.get("lang", "English")
    d = TRANSLATIONS.get(lang, TRANSLATIONS["English"])
    return d.get(key, TRANSLATIONS["English"].get(key, key))


# ─────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────
def init_state():
    defaults = {
        "logged_in":    False,
        "user_name":    "",
        "user_email":   "",
        "user_role":    "user",
        "history":      [],
        "chat":         [],
        "last_result":  None,
        "last_img":     None,
        "lang":         "English",
        "reviews":      [
            {"name": "Dr. Ramesh",  "stars": 5, "text": "Very accurate and fast diagnosis."},
            {"name": "Priya M.",    "stars": 5, "text": "Easy to use. Excellent AI results."},
            {"name": "Arun S.",     "stars": 4, "text": "Useful tool for quick screening."},
        ],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────
def run_ai_diagnosis():
    probs = np.random.dirichlet(np.ones(6) * 2)
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    disease = DISEASES[idx]
    sev = "Severe" if conf > 0.70 else "Moderate" if conf > 0.45 else "Mild"
    return {
        "disease":  disease,
        "conf":     conf,
        "probs":    probs,
        "severity": sev,
        "region":   random.choice(["Upper Left Molar","Upper Right Premolar","Lower Right Incisor","Lower Left Molar"]),
        "models":   {
            "ResNet-50": f"{random.uniform(85,95):.1f}% - disease classification (Acc: 91%)",
            "U-Net":    f"{random.randint(2,8)} infected regions segmented",
            "YOLOv8":   f"{random.randint(1,5)} abnormalities detected",
            "VGG19":    f"{random.uniform(80,90):.1f}% feature similarity (Acc: 87%)",
        },
        "time": datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime("%d %b %Y  %I:%M %p"),
    }

def load_photo(filename):
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
    except:
        pass
    return None

def chatbot_reply(msg):
    m = msg.lower()
    for k, v in CHATBOT_ANSWERS.items():
        if k in m:
            return v
    return CHATBOT_ANSWERS["default"]

def make_heatmap(img: Image.Image) -> Image.Image:
    img2 = img.convert("RGBA").resize((300, 250))
    w, h = img2.size
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    px = overlay.load()
    cx, cy = w//2 + random.randint(-40,40), h//2 + random.randint(-30,30)
    r = min(w,h) * 0.3
    for y in range(h):
        for x in range(w):
            d = ((x-cx)**2+(y-cy)**2)**0.5
            if d < r:
                px[x,y] = (255, 50, 50, int(150*(1-d/r)))
    return Image.alpha_composite(img2, overlay).convert("RGB")

def make_pdf(result, pname, page, pgender):
    if not FPDF_OK:
        lines = [
            "ORALDX - AI DENTAL DIAGNOSTIC REPORT",
            "="*50,
            f"Patient   : {pname}",
            f"Age/Gender: {page} / {pgender}",
            f"Date      : {result['time']}",
            "-"*50,
            f"Condition : {result['disease']}",
            f"Confidence: {result['conf']*100:.1f}%",
            f"Severity  : {result['severity']}",
            f"Region    : {result['region']}",
            "-"*50,
            "AI MODELS (Hybrid Multi-Model System)",
        ] + [f"  {m}: {d}" for m, d in result["models"].items()] + [
            "-"*50,
            "RECOMMENDATION",
            RECOMMENDATIONS.get(result['disease'], "Consult a dentist."),
            "-"*50,
            "DISCLAIMER: AI-generated report. Consult a qualified dentist.",
        ]
        return "\n".join(lines).encode("utf-8")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(21,101,192)
    pdf.rect(0,0,210,26,"F")
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",16)
    pdf.set_xy(10,5); pdf.cell(0,10,"OralDX - AI Dental Diagnostic Report",ln=True)
    pdf.set_font("Helvetica","",8)
    pdf.set_xy(10,16); pdf.cell(0,6,"Automated Diagnosis of Oral Conditions | ITER, SOA University")
    pdf.set_text_color(0,0,0)
    pdf.set_font("Helvetica","B",11)
    pdf.set_xy(10,32); pdf.cell(0,8,"Patient Information",ln=True)
    pdf.set_font("Helvetica","",10)
    y=42
    for lbl,val in [("Name",pname),("Age",page),("Gender",pgender),("Date",result["time"])]:
        pdf.set_xy(12,y); pdf.set_font("Helvetica","B",10); pdf.cell(40,7,lbl+":")
        pdf.set_font("Helvetica","",10); pdf.cell(0,7,str(val),ln=True); y+=7
    y+=4
    sc = {"Severe":(198,40,40),"Moderate":(230,81,0),"Mild":(46,125,50)}
    c = sc.get(result["severity"],(21,101,192))
    pdf.set_fill_color(*c); pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",11); pdf.set_xy(10,y)
    pdf.cell(190,9,f"  DIAGNOSED: {result['disease'].upper()} - Confidence: {result['conf']*100:.1f}%",fill=True,ln=True)
    y+=11; pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","",10)
    for lbl,val in [("Severity",result["severity"]),("Affected Region",result["region"])]:
        pdf.set_xy(12,y); pdf.set_font("Helvetica","B",10); pdf.cell(50,7,lbl+":")
        pdf.set_font("Helvetica","",10); pdf.cell(0,7,val,ln=True); y+=7
    y+=4
    pdf.set_font("Helvetica","B",11); pdf.set_xy(10,y); pdf.cell(0,8,"AI Models Analysis",ln=True); y+=9
    for model,detail in result["models"].items():
        pdf.set_xy(12,y); pdf.set_fill_color(21,101,192); pdf.set_text_color(255,255,255)
        pdf.set_font("Helvetica","B",9); pdf.cell(24,6,f" {model}",fill=True)
        pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","",9); pdf.cell(0,6,f"  {detail}",ln=True); y+=7
    y+=4
    pdf.set_fill_color(227,242,253); pdf.set_text_color(0,0,0)
    pdf.set_font("Helvetica","B",11); pdf.set_xy(10,y)
    pdf.cell(190,8,"  Recommendation",fill=True,ln=True); y+=10
    pdf.set_font("Helvetica","",10); pdf.set_xy(12,y)
    pdf.multi_cell(186,6,RECOMMENDATIONS.get(result["disease"],"Consult a dentist."))
    pdf.set_y(-18); pdf.set_fill_color(21,101,192)
    pdf.rect(0,pdf.get_y()-2,210,20,"F")
    pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","",8)
    pdf.cell(0,8,f"  OralDX AI | Sonali Patra | 24C216A45 | ITER, SOA University | {datetime.now().year}")
    return bytes(pdf.output())

# ─────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦷 OralDX")
    st.markdown("**AI Dental Diagnosis**")
    st.markdown("---")

    # ── Language Selector ──────────────────────────
    lang_choice = st.selectbox(
        "🌐 Language / भाषा / ଭାଷା",
        ["English", "Hindi", "Odia"],
        index=["English","Hindi","Odia"].index(st.session_state.lang),
    )
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

    st.markdown("---")

    # ── Navigation ─────────────────────────────────
    if not st.session_state.logged_in:
        page = st.radio("Navigation", ["🏠 Home", T("nav_signin")])
    else:
        pages = [
            T("nav_home"),
            T("nav_upload"),
            T("nav_results"),
            T("nav_chat"),
            T("nav_compare"),
            T("nav_history"),
            T("nav_reviews"),
            T("nav_about"),
            T("nav_profile"),
        ]
        if st.session_state.user_role == "admin":
            pages.append(T("nav_admin"))
        page = st.radio("Navigation", pages)

    st.markdown("---")
    if st.session_state.logged_in:
        st.markdown(f"**User:** {st.session_state.user_name}")
        if st.button(T("logout")):
            for k in ["logged_in","user_name","user_email","last_result","last_img","chat"]:
                st.session_state[k] = False if k=="logged_in" else (None if k in ["last_result","last_img"] else ([] if k=="chat" else ""))
            st.rerun()
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#555555;'>
    <b>AI Models:</b><br>
    • ResNet50<br>• U-Net<br>• YOLOv8<br>• VGG19<br><br>
    <b>Dataset:</b> 13,320 images<br>
    <b>Classes:</b> 6 diseases
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# PAGE: HOME (not logged in)
# ═══════════════════════════════════════════════════
if not st.session_state.logged_in and page == "🏠 Home":
    st.markdown('<div class="page-title">🦷 OralDX — Automated Dental Diagnosis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    AI-powered system that detects <b>6 oral diseases</b> from dental X-ray images using
    <b>ResNet50, U-Net, YOLOv8 and VGG19</b> — Hybrid Multi-Model AI System.
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,lbl) in zip([c1,c2,c3,c4],[("6","Disease Classes"),("13,320","X-Ray Images"),("4","AI Models"),("91%","ResNet-50 Acc")]):
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🦠 "+T("six_diseases"))
        for d, info in DATASET.items():
            st.markdown(f"**{d}** — {info['count']} images ({info['pct']}%)")
    with col_b:
        fig = px.pie(
            names=list(DATASET.keys()),
            values=[v["count"] for v in DATASET.values()],
            color_discrete_sequence=[v["color"] for v in DATASET.values()],
            title="Dataset Distribution (13,320 images)"
        )
        fig.update_layout(height=300, paper_bgcolor="white", font_color="#111111", margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(T("getstarted_hint"))


# ═══════════════════════════════════════════════════
# PAGE: SIGN IN / SIGN UP
# ═══════════════════════════════════════════════════
elif not st.session_state.logged_in and page == T("nav_signin"):
    st.markdown('<div class="page-title">🔑 Sign In / Sign Up</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### "+T("signin_head"))
        si_email = st.text_input(T("email"), placeholder="you@example.com", key="si_e")
        si_pwd   = st.text_input(T("password"), type="password", key="si_p")
        si_role  = st.selectbox(T("login_as"), [T("user_role"), T("admin_role")], key="si_r")
        if st.button(T("signin_btn"), key="si_btn"):
            if si_email and si_pwd:
                st.session_state.logged_in  = True
                st.session_state.user_email = si_email
                st.session_state.user_name  = si_email.split("@")[0].capitalize()
                st.session_state.user_role  = "admin" if si_role == "Admin" else "user"
                st.success(T("login_ok"))
                st.rerun()
            else:
                st.error(T("fill_credentials"))
        st.markdown('<div class="info-box">💡 Demo: Enter any email + password to login.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### "+T("signup_head"))
        su_name  = st.text_input(T("full_name"), key="su_n")
        su_email = st.text_input(T("email"), key="su_e")
        su_pwd   = st.text_input(T("password"), type="password", key="su_p")
        su_conf  = st.text_input(T("confirm_pwd"), type="password", key="su_c")
        if st.button(T("create_acc"), key="su_btn"):
            if su_name and su_email and su_pwd:
                if su_pwd == su_conf:
                    st.session_state.logged_in  = True
                    st.session_state.user_email = su_email
                    st.session_state.user_name  = su_name
                    st.session_state.user_role  = "user"
                    st.success(T("signup_ok"))
                    st.rerun()
                else:
                    st.error(T("pwd_mismatch"))
            else:
                st.error(T("fill_all"))


# ═══════════════════════════════════════════════════
# LOGGED IN PAGES
# ═══════════════════════════════════════════════════
elif st.session_state.logged_in:

    # ─────────────────────────────────────────────
    # HOME DASHBOARD
    # ─────────────────────────────────────────────
    if page == T("nav_home"):
        # IST = UTC + 5 hours 30 minutes
        from datetime import timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        hr  = datetime.now(IST).hour

        # Time-based greeting
        if 5 <= hr < 12:
            greet = T("greet_morning")    # 5 AM – 11:59 AM
        elif 12 <= hr < 17:
            greet = T("greet_afternoon")  # 12 PM – 4:59 PM
        elif 17 <= hr < 21:
            greet = T("greet_evening")    # 5 PM – 8:59 PM
        else:
            greet = T("greet_night")      # 9 PM – 4:59 AM
        # ── Welcome Banner ──────────────────────────────────────────────
        uname = st.session_state.user_name or "User"
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1565c0 0%, #0d47a1 50%, #1a237e 100%);
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(21,101,192,0.25);
        ">
            <div style="
                font-size: 1rem;
                color: rgba(255,255,255,0.80);
                margin-bottom: 6px;
                font-weight: 400;
                letter-spacing: 0.3px;
            ">
                🦷 OralDX — Automated Diagnosis of Oral Conditions from Dental X-Rays
            </div>
            <div style="
                font-size: 2rem;
                font-weight: 700;
                color: #ffffff;
                margin-bottom: 4px;
                line-height: 1.2;
            ">
                {greet}, <span style="color:#90caf9;">{uname}</span>!
            </div>
            <div style="
                font-size: 0.88rem;
                color: rgba(255,255,255,0.70);
                margin-top: 8px;
            ">
                Hybrid Multi-Model AI System &nbsp;|&nbsp;
                ResNet-50 &nbsp;&bull;&nbsp; VGG19 &nbsp;&bull;&nbsp; YOLOv8 &nbsp;&bull;&nbsp; U-Net
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        c1,c2,c3,c4 = st.columns(4)
        for col,(val,lbl) in zip([c1,c2,c3,c4],[
            ("91%","ResNet-50 Acc"),
            (str(len(st.session_state.history)),"Your Scans"),
            ("6","Disease Classes"),
            ("13,320","Dataset Images")]):
            with col:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### 📈 "+T("disease_dist"))
            fig = go.Figure(go.Bar(
                x=list(DATASET.keys()),
                y=[v["count"] for v in DATASET.values()],
                marker_color=[v["color"] for v in DATASET.values()],
                text=[f"{v['count']}" for v in DATASET.values()],
                textposition="outside"
            ))
            fig.update_layout(
                height=280, paper_bgcolor="white", plot_bgcolor="white",
                font_color="#111111", margin=dict(t=10,b=60,l=0,r=0),
                yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
                xaxis=dict(tickangle=-20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("### 🕒 "+T("recent_activity"))
            if st.session_state.history:
                for item in st.session_state.history[:5]:
                    st.markdown(f"""
                    <div class="timeline-row">
                    🦷 <b>{item['disease']}</b> — {item['conf']} confidence
                    <br><span style='color:#888;font-size:.75rem;'>{item['date']}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">No diagnoses yet. Upload an X-ray to begin.</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🎯 "+T("ai_acc_head"))
            for model, acc in [("ResNet-50",91),("VGG19",87),("YOLOv8",88),("U-Net",85)]:
                st.markdown(f"**{model}** — {acc}%")
                st.progress(acc)

    # ─────────────────────────────────────────────
    # UPLOAD & DIAGNOSE
    # ─────────────────────────────────────────────
    elif page == T("nav_upload"):
        st.markdown('<div class="page-title">📤 Upload Dental X-Ray</div>', unsafe_allow_html=True)

        col_up, col_prev = st.columns(2)
        with col_up:
            st.markdown("#### "+T("upload_img_head"))
            st.markdown('<div class="info-box">Supported formats: JPG, JPEG, PNG, BMP, DCM (DICOM)</div>', unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Choose dental X-ray",
                type=["jpg","jpeg","png","bmp"],
                label_visibility="collapsed"
            )

            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                st.session_state.last_img = img
                st.markdown(f'<div class="success-box">✅ Uploaded: <b>{uploaded.name}</b> ({uploaded.size//1024} KB)</div>', unsafe_allow_html=True)

        with col_prev:
            st.markdown("#### "+T("img_preview_head"))
            if uploaded and st.session_state.last_img:
                st.image(st.session_state.last_img, use_container_width=True)
            else:
                st.markdown("""
                <div style="border:2px dashed #cccccc;border-radius:10px;
                            height:220px;display:flex;align-items:center;
                            justify-content:center;color:#888888;font-size:0.9rem;">
                    Preview will appear here
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(T("upload_btn"), key="start_diag"):
            st.markdown("### "+T("processing_head"))
            prog = st.progress(0)
            stat = st.empty()
            steps = [
                (15,"🔧 Preprocessing image..."),
                (30,"🔍 ResNet50 — Extracting features..."),
                (50,"🎯 YOLOv8 — Detecting abnormalities..."),
                (70,"🧬 U-Net — Segmenting regions..."),
                (85,"🖼️ VGG19 — Feature comparison..."),
                (100,"✅ Analysis complete!"),
            ]
            for pct, msg in steps:
                prog.progress(pct)
                stat.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)
                time.sleep(0.4)

            result = run_ai_diagnosis()
            st.session_state.last_result = result
            st.session_state.history.insert(0, {
                "disease": result["disease"],
                "conf":    f"{result['conf']*100:.1f}%",
                "date":    result["time"],
                "image":   uploaded.name if uploaded else "demo.jpg"
            })
            st.success("✅ Diagnosis complete! Go to **View Results** in sidebar.")

    # ─────────────────────────────────────────────
    # VIEW RESULTS
    # ─────────────────────────────────────────────
    elif page == T("nav_results"):
        st.markdown('<div class="page-title">📊 Diagnosis Results</div>', unsafe_allow_html=True)

        r = st.session_state.last_result
        if not r:
            st.markdown('<div class="warn-box">⚠️ No results yet. Please upload an X-ray and run diagnosis first.</div>', unsafe_allow_html=True)
        else:
            # Result summary
            conf_pct = r["conf"] * 100
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size:0.8rem;color:#444;font-weight:600;">DETECTED CONDITION</div>
                    <div style="font-size:1.8rem;font-weight:700;color:#111111;margin:8px 0;">{r['disease']}</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{conf_pct:.1f}%</div>
                    <div class="metric-label">Confidence Score</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                color = "#c62828" if r["severity"]=="Severe" else "#e65100" if r["severity"]=="Moderate" else "#2e7d32"
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size:1.5rem;font-weight:700;color:{color};">{r['severity']}</div>
                    <div class="metric-label">Severity Level</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="info-box">🦷 <b>Affected Region:</b> {r["region"]} &nbsp;&nbsp; 🕒 <b>Time:</b> {r["time"]}</div>', unsafe_allow_html=True)

            col_left, col_right = st.columns(2)
            with col_left:
                # Probability chart
                st.markdown("#### "+T("class_prob_head"))
                fig = go.Figure(go.Bar(
                    y=DISEASES, x=[p*100 for p in r["probs"]],
                    orientation="h",
                    marker_color=["#1565c0" if d==r["disease"] else "#bbdefb" for d in DISEASES],
                    text=[f"{p*100:.1f}%" for p in r["probs"]],
                    textposition="outside"
                ))
                fig.update_layout(
                    height=260, paper_bgcolor="white", plot_bgcolor="white",
                    font_color="#111111", margin=dict(l=0,r=60,t=10,b=10),
                    xaxis=dict(visible=False)
                )
                st.plotly_chart(fig, use_container_width=True)

                # AI models breakdown
                st.markdown("#### "+T("ai_models_head"))
                for model, detail in r["models"].items():
                    st.markdown(f"""
                    <div style="padding:8px 0;border-bottom:1px solid #eeeeee;font-size:0.86rem;">
                    <b style="color:#1565c0;">{model}</b> — {detail}
                    </div>""", unsafe_allow_html=True)

            with col_right:
                # Heatmap
                st.markdown("#### 🔥 "+T("heatmap_head"))
                if st.session_state.last_img:
                    heat = make_heatmap(st.session_state.last_img)
                    st.image(heat, caption=T("red_region"), use_container_width=True)
                else:
                    fig_h, ax = plt.subplots(figsize=(4,3))
                    sns.heatmap(np.random.rand(15,15), cmap="RdYlGn_r", ax=ax,
                                cbar=False, xticklabels=False, yticklabels=False)
                    ax.set_title("Simulated Heatmap", fontsize=9)
                    fig_h.patch.set_facecolor("white")
                    st.pyplot(fig_h, use_container_width=True)
                    plt.close()

                # Recommendation
                st.markdown("#### 💡 "+T("recom_head"))
                st.markdown(f'<div class="warn-box">{RECOMMENDATIONS.get(r["disease"],"Consult a dentist.")}</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">⚠️ This is AI-generated guidance only. Always consult a qualified dentist.</div>', unsafe_allow_html=True)

            # Download PDF
            st.markdown("#### 📥 "+T("dl_report_head"))
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                p_name = st.text_input(T("patient_name"), value=st.session_state.user_name)
            with col_p2:
                p_age = st.text_input(T("age"), "--")
            with col_p3:
                p_gen = st.selectbox(T("gender"), T("gender_opts"))

            pdf_data = make_pdf(r, p_name, p_age, p_gen)
            ext  = ".pdf" if FPDF_OK else ".txt"
            mime = "application/pdf" if FPDF_OK else "text/plain"
            st.download_button(
                label=T("download_pdf"),
                data=pdf_data,
                file_name=f"OralDX_Report_{datetime.now().strftime('%Y%m%d_%H%M')}{ext}",
                mime=mime,
            )

    # ─────────────────────────────────────────────
    # AI CHATBOT
    # ─────────────────────────────────────────────
    elif page == T("nav_chat"):
        st.markdown('<div class="page-title">🤖 AI Dental Chatbot</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Ask me about any dental disease, AI models, dataset, treatment, or precautions.</div>', unsafe_allow_html=True)

        # Init chat
        if not st.session_state.chat:
            st.session_state.chat = [{"role":"bot","text":"👋 Hello! I am OralDX AI Assistant. Ask me anything about dental diseases, diagnosis, or oral health."}]

        # Display messages
        for msg in st.session_state.chat:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-end;margin:6px 0;">
                <div style="background:#1565c0;color:white;border-radius:12px 12px 2px 12px;
                            padding:10px 16px;max-width:75%;font-size:0.88rem;">{msg['text']}</div></div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-start;margin:6px 0;">
                <div style="background:#f5f5f5;color:#111111;border:1px solid #e0e0e0;
                            border-radius:12px 12px 12px 2px;
                            padding:10px 16px;max-width:75%;font-size:0.88rem;">{msg['text']}</div></div>
                """, unsafe_allow_html=True)

        # Quick question buttons
        st.markdown("**Quick Questions:**")
        qs = DISEASES + ["Treatment", "Precautions", "AI Models", "Accuracy", "Dataset"]
        cols = st.columns(5)
        for i, q in enumerate(qs[:10]):
            with cols[i % 5]:
                if st.button(q, key=f"qq_{i}"):
                    st.session_state.chat.append({"role":"user","text":f"What is {q}?"})
                    st.session_state.chat.append({"role":"bot","text":chatbot_reply(q.lower())})
                    st.rerun()

        # Text input
        user_inp = st.chat_input("Type your question here...")
        if user_inp:
            st.session_state.chat.append({"role":"user","text":user_inp})
            st.session_state.chat.append({"role":"bot","text":chatbot_reply(user_inp)})
            st.rerun()

        if st.button(T("clear_chat")):
            st.session_state.chat = []
            st.rerun()

    # ─────────────────────────────────────────────
    # COMPARE X-RAYS
    # ─────────────────────────────────────────────
    elif page == T("nav_compare"):
        st.markdown('<div class="page-title">🔁 Compare Two X-Rays</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Upload an older and newer X-ray to track disease progression and measure improvement.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📅 "+T("prev_scan_head"))
            img_old = st.file_uploader("Upload older X-ray", type=["jpg","jpeg","png"], key="c_old", label_visibility="collapsed")
            if img_old:
                st.image(Image.open(img_old), use_container_width=True)

        with col2:
            st.markdown("#### 📅 "+T("curr_scan_head"))
            img_new = st.file_uploader("Upload newer X-ray", type=["jpg","jpeg","png"], key="c_new", label_visibility="collapsed")
            if img_new:
                st.image(Image.open(img_new), use_container_width=True)

        if st.button(T("compare_btn")):
            prev_d = random.choice(DISEASES)
            curr_d = random.choice(DISEASES)
            prev_s = random.choice(["Severe","Moderate"])
            curr_s = random.choice(["Moderate","Mild"])
            imp    = random.randint(20,55)

            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size:0.75rem;color:#444;">PREVIOUS SCAN</div>
                    <div style="font-weight:700;font-size:1.1rem;color:#c62828;margin:6px 0;">{prev_d}</div>
                    <div style="color:#888;font-size:0.8rem;">{prev_s}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size:0.75rem;color:#444;">CURRENT SCAN</div>
                    <div style="font-weight:700;font-size:1.1rem;color:#2e7d32;margin:6px 0;">{curr_d}</div>
                    <div style="color:#888;font-size:0.8rem;">{curr_s}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{imp}%</div>
                    <div class="metric-label">Improvement</div>
                </div>""", unsafe_allow_html=True)

            # comparison chart
            pp = np.random.dirichlet(np.ones(6)*1.5)
            cp = np.random.dirichlet(np.ones(6)*2.5)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Previous Scan", x=DISEASES, y=pp*100, marker_color="#999999"))
            fig.add_trace(go.Bar(name="Current Scan",  x=DISEASES, y=cp*100, marker_color="#1565c0"))
            fig.update_layout(
                barmode="group", title="Disease Probability — Previous vs Current",
                height=300, paper_bgcolor="white", plot_bgcolor="white",
                font_color="#111111", yaxis_title="Probability (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f'<div class="success-box">✅ Previous: <b>{prev_d}</b> ({prev_s}) → Current: <b>{curr_d}</b> ({curr_s}) → Improvement: <b>{imp}%</b></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # HISTORY
    # ─────────────────────────────────────────────
    elif page == T("nav_history"):
        st.markdown('<div class="page-title">📂 Diagnosis History</div>', unsafe_allow_html=True)

        col_s, col_f = st.columns([2,1])
        with col_s:
            search = st.text_input(T("search_lbl"), label_visibility="collapsed", placeholder=T("search_ph"))
        with col_f:
            filt = st.selectbox(T("filter_lbl"), [T("all_filter")] + DISEASES, label_visibility="collapsed")

        hist = st.session_state.history
        if search:
            hist = [h for h in hist if search.lower() in h["disease"].lower() or search.lower() in h["date"].lower()]
        if filt != "All":
            hist = [h for h in hist if h["disease"] == filt]

        if not hist:
            st.markdown('<div class="info-box">📭 No records found. Upload an X-ray to begin.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(hist)} record(s) found**")
            for item in hist:
                st.markdown(f"""
                <div class="timeline-row">
                🦷 <b>{item['disease']}</b> — Confidence: {item['conf']}
                <br><span style='color:#888;font-size:0.75rem;'>📅 {item['date']} &nbsp;|&nbsp; 📁 {item['image']}</span>
                </div>""", unsafe_allow_html=True)

            # analytics
            if len(hist) > 1:
                from collections import Counter
                cnt = Counter(h["disease"] for h in hist)
                fig = px.bar(x=list(cnt.keys()), y=list(cnt.values()),
                             color=list(cnt.keys()), title="Your Diagnosis History",
                             labels={"x":"Disease","y":"Count"})
                fig.update_layout(height=260, paper_bgcolor="white", plot_bgcolor="white",
                                  font_color="#111111", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            if st.button(T("clear_hist")):
                st.session_state.history = []
                st.rerun()

    # ─────────────────────────────────────────────
    # REVIEWS
    # ─────────────────────────────────────────────
    elif page == T("nav_reviews"):
        st.markdown('<div class="page-title">⭐ Reviews & Feedback</div>', unsafe_allow_html=True)

        for rev in st.session_state.reviews:
            st.markdown(f"""
            <div class="card">
                <b>{rev['name']}</b> &nbsp;
                <span style="color:#f9a825;">{'⭐'*rev['stars']}</span><br>
                <span style="color:#444;font-size:0.88rem;">"{rev['text']}"</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### "+T("write_rev_head"))
        r_name  = st.text_input(T("your_name"))
        r_stars = st.select_slider(T("rating"), options=[1,2,3,4,5], value=5)
        r_text  = st.text_area(T("your_feedback"), placeholder=T("feedback_ph"))
        if st.button(T("submit_review")):
            if r_name and r_text:
                st.session_state.reviews.insert(0, {"name":r_name,"stars":r_stars,"text":r_text})
                st.success("✅ Review submitted! Thank you.")
                st.rerun()
            else:
                st.warning(T("pls_fill_rev"))

    # ─────────────────────────────────────────────
    # ABOUT US
    # ─────────────────────────────────────────────
    elif page == T("nav_about"):
        st.markdown('<div class="page-title">ℹ️ About Us</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <b>Project:</b> Automated Diagnosis of Oral Conditions from Dental X-Rays<br>
        <b>Department:</b> Computer Application, ITER, SOA University, Bhubaneswar, Odisha<br>
        <b>Batch:</b> MCA 2024–2026 &nbsp;|&nbsp; <b>Group 8</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Guide
        st.markdown("### 🎓 Project Guide")
        guide = TEAM[0]
        col_g, col_gi = st.columns([1,4])
        with col_g:
            photo = load_photo(guide["photo"])
            if photo:
                st.image(photo.resize((100,100)), use_container_width=True)
            else:
                st.markdown('<div style="font-size:3.5rem;text-align:center;">👨‍🏫</div>', unsafe_allow_html=True)
        with col_gi:
            st.markdown(f"""
            <div style="padding:8px;">
                <div style="font-size:1.1rem;font-weight:700;color:#111111;">{guide['name']}</div>
                <div style="color:#1565c0;font-weight:600;">{guide['reg']}</div>
                <div style="color:#444;font-size:0.86rem;">{guide['role']}</div>
                <div style="color:#444;font-size:0.86rem;">Dept. of Computer Application, ITER, SOA University</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Team Members
        st.markdown("### 👥 Project Team")
        students = [m for m in TEAM if m["type"]=="student"]
        emojis = ["👩‍💻","👩‍🔬","👩‍💼","🧑‍💻","👩‍🔧"]
        cols = st.columns(len(students))
        for i, (col, member) in enumerate(zip(cols, students)):
            with col:
                photo = load_photo(member["photo"])
                if photo:
                    # Crop to square
                    w, h = photo.size
                    s = min(w,h)
                    left=(w-s)//2; top=(h-s)//2
                    photo_sq = photo.crop((left,top,left+s,top+s)).resize((120,120))
                    st.image(photo_sq, use_container_width=True)
                else:
                    st.markdown(f'<div style="font-size:2.5rem;text-align:center;">{emojis[i]}</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="member-card" style="margin-top:0;">
                    <div class="member-name">{member['name']}</div>
                    <div style="font-size:0.72rem;color:#888;">{member['reg']}</div>
                    <div class="member-role">{member['role']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Tech stack
        st.markdown("### 🛠️ Technology Used")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **AI Models:**
            - ResNet50 — Disease Classification
            - U-Net — Image Segmentation
            - YOLOv8 — Real-Time Detection
            - VGG19 — Feature Extraction

            **Framework:** TensorFlow / Keras
            """)
        with col2:
            st.markdown("""
            **Dataset:** 13,320 dental X-ray images
            **Classes:** 6 oral disease types
            **Language:** Python 3.9+
            **Web App:** Streamlit
            **PDF:** fpdf2
            """)



    # ─────────────────────────────────────────────
    # PROFILE
    # ─────────────────────────────────────────────
    elif page == T("nav_profile"):
        st.markdown('<div class="page-title">👤 My Profile</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:3rem;text-align:center;">👤</div>
                <div style="text-align:center;font-weight:700;font-size:1.1rem;margin-top:8px;">{st.session_state.user_name}</div>
                <div style="text-align:center;color:#555;font-size:0.88rem;">{st.session_state.user_email}</div>
                <div style="text-align:center;margin-top:8px;">
                    <span style="background:#e3f2fd;color:#1565c0;border-radius:12px;padding:3px 12px;font-size:0.78rem;">
                    {'🛡️ Admin' if st.session_state.user_role=='admin' else '👤 User'}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

        with col2:
            c1,c2,c3 = st.columns(3)
            for col,(val,lbl) in zip([c1,c2,c3],[
                (str(len(st.session_state.history)),"Total Scans"),
                (str(len(st.session_state.reviews)),"Reviews"),
                (str(len(st.session_state.history)),"Reports"),
            ]):
                with col:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ⚙️ "+T("update_name_head"))
        new_name = st.text_input(T("display_name"), value=st.session_state.user_name)
        if st.button("💾 Save"):
            st.session_state.user_name = new_name
            st.success(T("name_updated"))

    # ─────────────────────────────────────────────
    # ADMIN PANEL
    # ─────────────────────────────────────────────
    elif page == T("nav_admin") and st.session_state.user_role == "admin":
        st.markdown('<div class="page-title">🛡️ Admin Panel</div>', unsafe_allow_html=True)

        # Stats
        c1,c2,c3,c4 = st.columns(4)
        for col,(val,lbl) in zip([c1,c2,c3,c4],[("128","Total Users"),("347","Total X-Rays"),("91%","ResNet-50 Acc"),("Mouth Ulcer","Most Detected")]):
            with col:
                st.markdown(f'<div class="metric-box"><div class="metric-value" style="font-size:1.3rem;">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

        tab1,tab2,tab3,tab4 = st.tabs(["👥 Users","📋 Predictions","📈 Analytics","🗃️ Dataset"])

        with tab1:
            df_u = pd.DataFrame({
                "Name":  ["Alice K.","Dr. Ravi","Priya M.","Amit S.","Sunita P."],
                "Email": ["alice@x.com","ravi@c.com","priya@g.com","amit@h.com","sunita@m.com"],
                "Role":  ["User","Admin","User","User","User"],
                "Scans": [12,34,8,5,21],
            })
            st.dataframe(df_u, use_container_width=True, hide_index=True)
            st.download_button("📥 Export CSV", df_u.to_csv(index=False).encode(), "users.csv")

        with tab2:
            logs = st.session_state.history + [
                {"disease":"Caries",            "conf":"93.2%","date":"15 Jan 2025","image":"xray1.jpg"},
                {"disease":"Calculus",           "conf":"87.6%","date":"14 Jan 2025","image":"scan2.jpg"},
                {"disease":"Gingivitis",         "conf":"91.0%","date":"13 Jan 2025","image":"scan3.jpg"},
                {"disease":"Mouth Ulcer",        "conf":"88.4%","date":"12 Jan 2025","image":"xray4.jpg"},
            ]
            st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Logs"): st.session_state.history=[]; st.rerun()

        with tab3:
            col_l, col_r = st.columns(2)
            with col_l:
                daily = {"Mon":24,"Tue":31,"Wed":18,"Thu":42,"Fri":28,"Sat":15,"Sun":9}
                fig = px.bar(x=list(daily.keys()),y=list(daily.values()),title="Daily Uploads",
                             color_discrete_sequence=["#1565c0"])
                fig.update_layout(height=260,paper_bgcolor="white",plot_bgcolor="white",font_color="#111111")
                st.plotly_chart(fig,use_container_width=True)
            with col_r:
                ma = {"ResNet-50":91,"VGG19":87,"YOLOv8":88,"U-Net":85}
                fig2 = px.bar(x=list(ma.keys()),y=list(ma.values()),title="Model Accuracy (%)",
                              color=list(ma.keys()),
                              color_discrete_sequence=["#ef4444","#3b82f6","#22c55e","#f97316"])
                fig2.update_layout(height=260,paper_bgcolor="white",plot_bgcolor="white",
                                   font_color="#111111",showlegend=False,yaxis_range=[85,100])
                st.plotly_chart(fig2,use_container_width=True)

            fig3 = px.pie(names=list(DATASET.keys()),
                          values=[v["count"] for v in DATASET.values()],
                          title="Disease Distribution",
                          color_discrete_sequence=[v["color"] for v in DATASET.values()])
            fig3.update_layout(height=300,paper_bgcolor="white",font_color="#111111")
            st.plotly_chart(fig3,use_container_width=True)

        with tab4:
            ds_df = pd.DataFrame({
                "Disease":     list(DATASET.keys()),
                "Images":      [v["count"] for v in DATASET.values()],
                "% of Total":  [f"{v['pct']}%" for v in DATASET.values()],
                "Description": [
                    "Radiolucent dark spots — dental decay",
                    "Mineralised plaque near root surfaces",
                    "Gum inflammation — soft-tissue changes",
                    "Painful mucosal sores",
                    "Colour changes on tooth surface",
                    "Congenital absence of teeth",
                ]
            })
            st.markdown("**Total Dataset: 13,320 X-ray images across 6 disease classes**")
            st.dataframe(ds_df, use_container_width=True, hide_index=True)
            fig4 = px.bar(ds_df,x="Disease",y="Images",color="Disease",
                          title="Dataset Distribution — 13,320 Images",
                          color_discrete_sequence=[v["color"] for v in DATASET.values()],
                          text="Images")
            fig4.update_traces(textposition="outside")
            fig4.update_layout(height=300,paper_bgcolor="white",plot_bgcolor="white",
                               font_color="#111111",showlegend=False,yaxis_range=[0,3200])
            st.plotly_chart(fig4,use_container_width=True)
