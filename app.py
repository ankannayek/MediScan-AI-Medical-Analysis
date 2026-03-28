from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')
app = Flask(__name__, static_folder="assets")

# ─── Load Symptom Model ───────────────────────────────────────────────────────
with open('Disease/symptoms_list.pkl', 'rb') as f:
    SYMPTOMS = pickle.load(f)
with open('Disease/disease_model.pkl', 'rb') as f:
    RF_MODEL = pickle.load(f)
CLASSES = list(RF_MODEL.classes_)

# ─── Load Keras Models (requires tensorflow) ─────────────────────────────────
FRACTURE_MODEL     = None
DIAGNOSTIC_MODEL   = None
SKIN_DISEASE_MODEL = None

# Scaler params extracted from scaler.pkl (StandardScaler fitted on training data)
# Matches 9 input features: blood_glucose, hba1c, sys_bp, dia_bp, ldl, hdl, triglycerides, haemoglobin, mcv
_BLOOD_SCALER_MEAN  = [102.8842, 5.5106, 112.6776, 72.1072, 100.4365, 48.8877, 126.4954, 13.8172, 87.3571]
_BLOOD_SCALER_SCALE = [ 26.8585, 1.0916,  16.3704,  9.6164,  34.9007,  7.4056,  65.9189,  1.5208,  5.7416]

def blood_scale(vec):
    arr = np.array(vec, dtype=np.float32)
    return ((arr - _BLOOD_SCALER_MEAN) / _BLOOD_SCALER_SCALE).astype(np.float32)

# ─── Keras 3 compatibility loader ─────────────────────────────────────────────
# Keras 3.x saves .keras files with InputLayer config keys (e.g. 'optional')
# that older TF/Keras versions cannot deserialize. This patches config.json
# inside the zip before loading, stripping unrecognised keys.
def _load_keras_model_compat(keras_path):
    import zipfile, json, tempfile, shutil
    import tensorflow as tf

    with zipfile.ZipFile(keras_path, 'r') as zin:
        config = json.loads(zin.read('config.json'))

    _INPUTLAYER_ALLOWED = {
        'batch_shape', 'batch_input_shape', 'dtype',
        'sparse', 'ragged', 'name'
    }
    for layer in config.get('config', {}).get('layers', []):
        if layer.get('class_name') == 'InputLayer':
            lc = layer.get('config', {})
            for k in [k for k in list(lc.keys()) if k not in _INPUTLAYER_ALLOWED]:
                del lc[k]

    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, 'model_compat.keras')
    try:
        with zipfile.ZipFile(keras_path, 'r') as zin, \
             zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == 'config.json':
                    zout.writestr(item, json.dumps(config))
                else:
                    zout.writestr(item, zin.read(item.filename))
        model = tf.keras.models.load_model(tmp_path, compile=False)
        return model
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ─── Load Fracture Model ──────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    FRACTURE_MODEL = tf.keras.models.load_model('Bone/keras_model.h5', compile=False)
    print("✅ Fracture model loaded")
except Exception as e:
    print(f"⚠️  Fracture model not loaded: {e}")

# ─── Load Blood Diagnostic Model ──────────────────────────────────────────────
# Architecture: Input(9) → Dense(32,relu) → Dense(16,relu) → Dense(5,softmax)
# Loss: sparse_categorical_crossentropy
# Classes (index 0–4): Anemia, Diabetes, Fit/Healthy, High Cholesterol, Hypertension
# ─── Load Blood Diagnostic Model (HuggingFace) ────────────────────────────────
try:
    HF_MODEL_PATH = "Blood"

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
    DIAGNOSTIC_MODEL = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_PATH)

    DIAGNOSTIC_MODEL.eval()
    print("✅ HuggingFace Diagnostic model loaded")

except Exception as e:
    DIAGNOSTIC_MODEL = None
    print(f"⚠️  Diagnostic model not loaded: {e}")

# ─── Load Skin Disease Model ──────────────────────────────────────────────────
try:
    if 'tf' not in dir():
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    SKIN_DISEASE_MODEL = tf.keras.models.load_model('Skin/keras_model.h5', compile=False)
    print("✅ Skin disease model loaded")
except Exception as e:
    print(f"⚠️  Skin disease model not loaded: {e}")

# ─── Disease info ─────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Fungal infection": {"severity": "low", "icon": "🦠", "advice": "Keep affected areas dry and clean. Antifungal creams may help."},
    "Allergy": {"severity": "low", "icon": "🤧", "advice": "Identify and avoid allergens. Antihistamines may provide relief."},
    "GERD": {"severity": "medium", "icon": "🔥", "advice": "Avoid spicy foods, eat smaller meals, and don't lie down after eating."},
    "Chronic cholestasis": {"severity": "high", "icon": "⚠️", "advice": "Consult a gastroenterologist immediately for liver function tests."},
    "Drug Reaction": {"severity": "high", "icon": "💊", "advice": "Stop the suspected medication and seek medical attention immediately."},
    "Peptic ulcer diseae": {"severity": "medium", "icon": "🫃", "advice": "Avoid NSAIDs, alcohol, and spicy food. Consult a doctor for medication."},
    "AIDS": {"severity": "high", "icon": "🔴", "advice": "Seek immediate medical care for antiretroviral therapy and counseling."},
    "Diabetes ": {"severity": "high", "icon": "🩸", "advice": "Monitor blood glucose, maintain diet, and consult an endocrinologist."},
    "Gastroenteritis": {"severity": "medium", "icon": "🤢", "advice": "Stay hydrated with oral rehydration salts. Rest and eat bland foods."},
    "Bronchial Asthma": {"severity": "medium", "icon": "🫁", "advice": "Use prescribed inhalers, avoid triggers, and see a pulmonologist."},
    "Hypertension ": {"severity": "high", "icon": "❤️", "advice": "Reduce salt intake, exercise regularly, and take prescribed medication."},
    "Migraine": {"severity": "medium", "icon": "🧠", "advice": "Rest in a dark quiet room, stay hydrated, and consult a neurologist."},
    "Cervical spondylosis": {"severity": "medium", "icon": "🦴", "advice": "Do physiotherapy exercises and consult an orthopedist."},
    "Paralysis (brain hemorrhage)": {"severity": "critical", "icon": "🚨", "advice": "EMERGENCY: Call ambulance immediately. This is a life-threatening condition."},
    "Jaundice": {"severity": "high", "icon": "🟡", "advice": "Rest, drink plenty of fluids, and consult a hepatologist urgently."},
    "Malaria": {"severity": "high", "icon": "🦟", "advice": "Seek immediate medical treatment. Antimalarial drugs are needed."},
    "Chicken pox": {"severity": "low", "icon": "🔴", "advice": "Rest, avoid scratching, use calamine lotion, and isolate yourself."},
    "Dengue": {"severity": "high", "icon": "🦟", "advice": "Seek immediate hospital care. Monitor platelet count carefully."},
    "Typhoid": {"severity": "high", "icon": "🌡️", "advice": "Take prescribed antibiotics. Drink clean water and avoid raw foods."},
    "Hepatitis A": {"severity": "medium", "icon": "🔶", "advice": "Rest, avoid alcohol, eat a healthy diet, and practice good hygiene."},
    "Hepatitis B": {"severity": "high", "icon": "🔶", "advice": "Consult a hepatologist for antiviral treatment and monitoring."},
    "Hepatitis C": {"severity": "high", "icon": "🔶", "advice": "Seek specialist care for direct-acting antiviral therapy."},
    "Hepatitis D": {"severity": "high", "icon": "🔶", "advice": "Requires hepatitis B treatment too. Consult a liver specialist urgently."},
    "Hepatitis E": {"severity": "medium", "icon": "🔶", "advice": "Rest, stay hydrated, and avoid alcohol. Usually resolves on its own."},
    "Alcoholic hepatitis": {"severity": "high", "icon": "🍺", "advice": "Stop alcohol immediately and seek medical care for liver evaluation."},
    "Tuberculosis": {"severity": "high", "icon": "😷", "advice": "Start DOTS therapy immediately. Isolate yourself to prevent spread."},
    "Common Cold": {"severity": "low", "icon": "🤧", "advice": "Rest, stay hydrated, use saline nasal spray, and take OTC medications."},
    "Pneumonia": {"severity": "high", "icon": "🫁", "advice": "Seek immediate medical care. Antibiotics and hospitalization may be needed."},
    "Dimorphic hemmorhoids(piles)": {"severity": "medium", "icon": "🔴", "advice": "Increase fiber intake, stay hydrated, and consult a proctologist."},
    "Heart attack": {"severity": "critical", "icon": "💔", "advice": "EMERGENCY: Call ambulance immediately. Chew aspirin if not allergic."},
    "Varicose veins": {"severity": "low", "icon": "🦵", "advice": "Wear compression stockings, elevate legs, and avoid prolonged standing."},
    "Hypothyroidism": {"severity": "medium", "icon": "🦋", "advice": "Take prescribed thyroid hormone replacement. Get regular TSH tests."},
    "Hyperthyroidism": {"severity": "medium", "icon": "🦋", "advice": "Consult an endocrinologist for antithyroid medication or therapy."},
    "Hypoglycemia": {"severity": "high", "icon": "🩸", "advice": "Immediately consume sugar. Monitor blood glucose levels regularly."},
    "Osteoarthristis": {"severity": "medium", "icon": "🦴", "advice": "Physical therapy, pain relief medications, and weight management help."},
    "Arthritis": {"severity": "medium", "icon": "🦴", "advice": "Anti-inflammatory medications, physiotherapy, and gentle exercise."},
    "(vertigo) Paroymsal  Positional Vertigo": {"severity": "medium", "icon": "💫", "advice": "Epley maneuver can help. Avoid sudden head movements."},
    "Acne": {"severity": "low", "icon": "😤", "advice": "Keep skin clean, use non-comedogenic products, and consult a dermatologist."},
    "Urinary tract infection": {"severity": "medium", "icon": "💧", "advice": "Drink plenty of water and take prescribed antibiotics."},
    "Psoriasis": {"severity": "medium", "icon": "🔴", "advice": "Use moisturizers, prescribed topical treatments, and avoid triggers."},
    "Impetigo": {"severity": "low", "icon": "🦠", "advice": "Apply antibiotic ointment and keep the area clean. Avoid contact."},
    "hepatitis A": {"severity": "medium", "icon": "🔶", "advice": "Rest, avoid alcohol, eat a healthy diet, and practice good hygiene."},
}

# ─── Skin disease classes (ResNet-50 → 23 classes) ────────────────────────────
SKIN_DISEASE_CLASSES = [
    {"label": "Warts & Molluscum",                          "icon": "🟡", "severity": "low",      "advice": "Viral skin growths. Salicylic acid or cryotherapy are effective. Usually resolves on its own."},
    {"label": "Vasculitis",                                 "icon": "🟣", "severity": "high",     "advice": "Inflammation of blood vessels. Requires urgent medical evaluation and systemic treatment."},
    {"label": "Vascular Tumors",                            "icon": "🟣", "severity": "medium",   "advice": "Includes haemangiomas and other vascular growths. Consult a dermatologist for evaluation."},
    {"label": "Urticaria / Hives",                          "icon": "🟠", "severity": "medium",   "advice": "Allergic skin reaction. Antihistamines help. Seek emergency care if throat swelling occurs."},
    {"label": "Tinea / Ringworm / Candidiasis",             "icon": "🟡", "severity": "low",      "advice": "Fungal infection. Antifungal creams (clotrimazole, terbinafine) usually resolve it in 2–4 weeks."},
    {"label": "Systemic Disease",                           "icon": "🔴", "severity": "high",     "advice": "Skin symptoms may reflect an underlying systemic condition. See a doctor for full evaluation."},
    {"label": "Seborrheic Keratoses",                       "icon": "🟤", "severity": "low",      "advice": "Non-cancerous skin growth. Usually harmless; removal is cosmetic. Monitor for changes."},
    {"label": "Scabies / Lyme Disease",                     "icon": "🟠", "severity": "high",     "advice": "Parasitic or tick-borne infection. Requires prescription treatment. See a doctor promptly."},
    {"label": "Psoriasis / Lichen Planus",                  "icon": "🟠", "severity": "medium",   "advice": "Chronic inflammatory skin condition. Topical treatments and biologics can help. See a dermatologist."},
    {"label": "Poison Ivy / Contact Dermatitis",            "icon": "🟠", "severity": "medium",   "advice": "Allergic reaction to plant contact. Wash the area, apply hydrocortisone, and take antihistamines."},
    {"label": "Nail Fungus",                                "icon": "🟡", "severity": "low",      "advice": "Fungal nail infection. Oral or topical antifungals needed. Treatment takes several months."},
    {"label": "Melanoma / Skin Cancer / Nevi & Moles",      "icon": "🔴", "severity": "critical", "advice": "URGENT: Possible skin cancer. See a dermatologist or oncologist immediately for biopsy."},
    {"label": "Lupus & Connective Tissue Disease",          "icon": "🟣", "severity": "high",     "advice": "Autoimmune condition. Requires rheumatology evaluation and long-term systemic treatment."},
    {"label": "Pigmentation Disorders",                     "icon": "⚪", "severity": "low",      "advice": "Light-related or pigmentation disorder. Consult a dermatologist for diagnosis and management."},
    {"label": "Herpes / HPV / STDs",                        "icon": "🟠", "severity": "high",     "advice": "Sexually transmitted infection. Seek confidential medical care for antiviral or other treatment."},
    {"label": "Hair Loss / Alopecia",                       "icon": "🟤", "severity": "medium",   "advice": "Hair loss can have many causes. See a dermatologist for diagnosis and treatment options."},
    {"label": "Exanthems & Drug Eruptions",                 "icon": "🟠", "severity": "high",     "advice": "May indicate a drug reaction or viral rash. Stop suspected medication and see a doctor urgently."},
    {"label": "Eczema / Atopic Dermatitis",                 "icon": "🟠", "severity": "medium",   "advice": "Inflammatory skin condition. Use moisturisers, avoid triggers, and consult a dermatologist."},
    {"label": "Cellulitis / Impetigo",                      "icon": "🔴", "severity": "high",     "advice": "Bacterial skin infection. Requires topical or oral antibiotics. See a doctor promptly."},
    {"label": "Bullous Disease",                            "icon": "🔴", "severity": "high",     "advice": "Blistering skin disorder. Requires urgent dermatology evaluation and systemic treatment."},
    {"label": "Atopic Dermatitis",                          "icon": "🟠", "severity": "medium",   "advice": "Chronic eczema. Regular moisturising, topical steroids, and avoiding triggers help manage it."},
    {"label": "Actinic Keratosis / Basal Cell Carcinoma",   "icon": "🔴", "severity": "high",     "advice": "Pre-cancerous or cancerous lesion from sun damage. See a dermatologist for prompt treatment."},
    {"label": "Acne & Rosacea",                             "icon": "🟡", "severity": "low",      "advice": "Common skin conditions. Benzoyl peroxide, retinoids, or antibiotics can help. See a dermatologist."},
]

BLOOD_FEATURES = [
    {"key": "blood_glucose", "label": "Blood Glucose",    "unit": "mg/dL",  "min": 50,  "max": 400, "default": 103.0, "step": 1},
    {"key": "hba1c",         "label": "HbA1C",            "unit": "%",      "min": 3.5, "max": 15,  "default": 5.5,   "step": 0.1},
    {"key": "sys_bp",        "label": "Systolic BP",      "unit": "mmHg",   "min": 80,  "max": 200, "default": 113.0, "step": 1},
    {"key": "dia_bp",        "label": "Diastolic BP",     "unit": "mmHg",   "min": 40,  "max": 130, "default": 72.0,  "step": 1},
    {"key": "ldl",           "label": "LDL Cholesterol",  "unit": "mg/dL",  "min": 30,  "max": 300, "default": 100.0, "step": 1},
    {"key": "hdl",           "label": "HDL Cholesterol",  "unit": "mg/dL",  "min": 20,  "max": 100, "default": 49.0,  "step": 1},
    {"key": "triglycerides", "label": "Triglycerides",    "unit": "mg/dL",  "min": 30,  "max": 500, "default": 126.0, "step": 1},
    {"key": "haemoglobin",   "label": "Haemoglobin",      "unit": "g/dL",   "min": 5,   "max": 20,  "default": 13.8,  "step": 0.1},
    {"key": "mcv",           "label": "MCV",              "unit": "fL",     "min": 50,  "max": 120, "default": 87.0,  "step": 0.5},
]

# ─── Diagnostic classes — index MUST match model output order (0–4) ───────────
DIAGNOSTIC_CLASSES = [
    {"label": "Anemia",           "icon": "🩸", "severity": "medium", "advice": "Low haemoglobin detected. Consult a doctor for iron, B12, or folate supplementation."},
    {"label": "Diabetes",         "icon": "🍬", "severity": "high",   "advice": "Elevated glucose/HbA1C detected. See an endocrinologist for glucose management."},
    {"label": "Fit / Healthy",    "icon": "✅", "severity": "low",    "advice": "Your blood panel looks normal. Maintain a healthy lifestyle and schedule regular checkups."},
    {"label": "High Cholesterol", "icon": "🫀", "severity": "high",   "advice": "Elevated LDL or triglycerides detected. Reduce saturated fats and consult a cardiologist."},
    {"label": "Hypertension",     "icon": "❤️", "severity": "high",   "advice": "Elevated blood pressure detected. Reduce salt intake, exercise regularly, and see a doctor."},
]

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'symptom_model':      True,
        'fracture_model':     FRACTURE_MODEL is not None,
        'diagnostic_model':   DIAGNOSTIC_MODEL is not None,
        'skin_disease_model': SKIN_DISEASE_MODEL is not None,
    })

@app.route('/api/symptoms')
def get_symptoms():
    formatted = [s.replace('_', ' ').strip().title() for s in SYMPTOMS]
    return jsonify({'symptoms': SYMPTOMS, 'formatted': formatted})

@app.route('/api/predict/symptoms', methods=['POST'])
def predict_symptoms():
    data = request.json
    selected = data.get('symptoms', [])
    if not selected:
        return jsonify({'error': 'No symptoms provided'}), 400
    vec = np.zeros(len(SYMPTOMS))
    for sym in selected:
        if sym in SYMPTOMS:
            vec[SYMPTOMS.index(sym)] = 1
    proba = RF_MODEL.predict_proba([vec])[0]
    top5  = np.argsort(proba)[::-1][:5]
    results = []
    for i in top5:
        disease = CLASSES[i]
        conf    = float(proba[i])
        if conf > 0.01:
            info = DISEASE_INFO.get(disease, {"severity": "medium", "icon": "🏥", "advice": "Please consult a healthcare professional."})
            results.append({'disease': disease, 'confidence': round(conf * 100, 1),
                            'severity': info['severity'], 'icon': info['icon'], 'advice': info['advice']})
    return jsonify({'results': results})

@app.route('/api/predict/fracture', methods=['POST'])
def predict_fracture():
    if FRACTURE_MODEL is None:
        return jsonify({'error': 'Fracture model not available. Install TensorFlow to enable this feature.'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        from PIL import Image
        file = request.files['image']
        img  = Image.open(file.stream).convert('RGB').resize((224, 224))
        arr  = np.array(img, dtype=np.float32) / 255.0
        arr  = np.expand_dims(arr, axis=0)
        preds  = FRACTURE_MODEL.predict(arr, verbose=0)[0]
        labels = ['Normal', 'Fracture']
        top    = int(np.argmax(preds))
        return jsonify({
            'label':       labels[top],
            'confidence':  round(float(preds[top]) * 100, 1),
            'is_fracture': top == 1,
            'scores':      {labels[i]: round(float(preds[i]) * 100, 1) for i in range(len(preds))}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/blood-features')
def blood_features():
    return jsonify({'features': BLOOD_FEATURES, 'classes': DIAGNOSTIC_CLASSES})

@app.route('/api/predict/diagnostic', methods=['POST'])
def predict_diagnostic():
    if DIAGNOSTIC_MODEL is None:
        return jsonify({'error': 'Diagnostic model not available. Install model.'}), 503

    data = request.json
    try:
        raw = {
            "blood_glucose": float(data.get('blood_glucose', 103.0)),
            "hba1c": float(data.get('hba1c', 5.5)),
            "sys_bp": float(data.get('sys_bp', 113.0)),
            "dia_bp": float(data.get('dia_bp', 72.0)),
            "ldl": float(data.get('ldl', 100.0)),
            "hdl": float(data.get('hdl', 49.0)),
            "triglycerides": float(data.get('triglycerides', 126.0)),
            "haemoglobin": float(data.get('haemoglobin', 13.8)),
            "mcv": float(data.get('mcv', 87.0)),
        }

        # 👉 SAME FORMAT AS TRAINING
        text = " ".join([f"{k}:{v}" for k, v in raw.items()])

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = DIAGNOSTIC_MODEL(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()

        if len(probs) != len(DIAGNOSTIC_CLASSES):
            return jsonify({'error': 'Model output size mismatch'}), 500

        top = int(np.argmax(probs))

        results = []
        for i, score in enumerate(probs):
            cls = DIAGNOSTIC_CLASSES[i]
            results.append({
                'label': cls['label'],
                'icon': cls['icon'],
                'severity': cls['severity'],
                'advice': cls['advice'],
                'confidence': round(float(score) * 100, 1),
                'is_top': bool(i == top),
            })

        results.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/skin-disease', methods=['POST'])
def predict_skin_disease():
    if SKIN_DISEASE_MODEL is None:
        return jsonify({'error': 'Skin disease model not available. Install TensorFlow to enable this feature.'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        from PIL import Image
        file     = request.files['image']
        img      = Image.open(file.stream).convert('RGB').resize((224, 224))
        arr      = np.array(img, dtype=np.float32) / 255.0
        arr      = np.expand_dims(arr, axis=0)
        preds    = SKIN_DISEASE_MODEL.predict(arr, verbose=0)[0]
        top5_idx = np.argsort(preds)[::-1][:5]
        results  = []
        for i in top5_idx:
            cls = SKIN_DISEASE_CLASSES[i] if i < len(SKIN_DISEASE_CLASSES) else \
                  {"label": f"Class {i}", "icon": "🔬", "severity": "medium", "advice": "Consult a dermatologist."}
            results.append({
                'label':      cls['label'],
                'icon':       cls['icon'],
                'severity':   cls['severity'],
                'advice':     cls['advice'],
                'confidence': round(float(preds[i]) * 100, 1),
                'is_top':     bool(i == top5_idx[0]),
            })
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
