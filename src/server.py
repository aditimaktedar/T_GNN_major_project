import sqlite3
import os
import sys
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.portal_adapter import orchestrator_regimen_to_portal
from api.service import ClinicalDDIOrchestrator

DB_PATH = os.path.join(PROJECT_ROOT, "data", "clinical_ddi.db")
_orchestrator: ClinicalDDIOrchestrator | None = None


def get_orchestrator() -> ClinicalDDIOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ClinicalDDIOrchestrator.get_instance()
    return _orchestrator


def _medication_lookup_key(drug: dict) -> str | None:
    drugbank_id = drug.get("drugbank_id")
    if drugbank_id:
        return str(drugbank_id)
    alt_id = drug.get("id")
    if isinstance(alt_id, str) and alt_id.upper().startswith("DB"):
        return alt_id
    name = drug.get("name")
    return str(name).strip() if name else None


def _normalize_medication_row(row: dict) -> dict:
    """Expose DrugBank ID as ``id`` for the portal while keeping SQLite PK as ``db_id``."""
    normalized = dict(row)
    normalized["db_id"] = normalized.get("id")
    if normalized.get("drugbank_id"):
        normalized["id"] = normalized["drugbank_id"]
    return normalized

app = Flask(__name__)
CORS(app)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Ensure DB initialized on startup
if not os.path.exists(DB_PATH):
    from src.init_db import init_db
    init_db()

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        engine = get_orchestrator().health()
    except Exception as exc:
        engine = {"status": "degraded", "error": str(exc)}
    return jsonify({
        "status": "ok",
        "database": DB_PATH,
        "engine": engine,
    })

@app.route('/api/patient', methods=['GET'])
def get_patient():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patients LIMIT 1;")
    p_row = cursor.fetchone()
    if not p_row:
        conn.close()
        return jsonify({"error": "No patient record found"}), 404

    patient = dict(p_row)

    cursor.execute("SELECT id, drugbank_id, name, dosage, frequency, route FROM medications WHERE status = 'Active';")
    m_rows = cursor.fetchall()
    medications = [_normalize_medication_row(dict(r)) for r in m_rows]
    patient['medications'] = medications

    conn.close()
    return jsonify(patient)

@app.route('/api/patient', methods=['PUT'])
def update_patient():
    data = request.json or {}
    conn = get_db()
    cursor = conn.cursor()

    name = data.get('name', 'Eleanor Vance')
    age = data.get('age', 72)
    gender = data.get('gender', 'Female')

    cursor.execute("""
    UPDATE patients SET name = ?, age = ?, gender = ? WHERE id = (SELECT id FROM patients LIMIT 1);
    """, (name, age, gender))
    conn.commit()

    cursor.execute("SELECT * FROM patients LIMIT 1;")
    patient = dict(cursor.fetchone())
    cursor.execute("SELECT id, drugbank_id, name, dosage, frequency, route FROM medications WHERE status = 'Active';")
    patient['medications'] = [_normalize_medication_row(dict(r)) for r in cursor.fetchall()]

    conn.close()
    return jsonify(patient)

@app.route('/api/medications', methods=['POST'])
def add_medication():
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({"error": "Drug name is required"}), 400

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM medications WHERE status = 'Active';")
    count = cursor.fetchone()[0]
    if count >= 15:
        conn.close()
        return jsonify({"error": "Maximum T-GNN capacity of 15 medications reached"}), 400

    drugbank_id = data.get('drugbank_id') or data.get('id') or f"DB{hash(name) % 90000 + 10000}"
    dosage = data.get('dosage', '10 mg')
    frequency = data.get('frequency', 'QD (Daily)')
    route = data.get('route', 'Oral')

    cursor.execute("""
    INSERT INTO medications (patient_id, drugbank_id, name, dosage, frequency, route)
    VALUES ((SELECT id FROM patients LIMIT 1), ?, ?, ?, ?, ?);
    """, (drugbank_id, name, dosage, frequency, route))
    conn.commit()

    new_id = cursor.lastrowid
    cursor.execute("SELECT id, drugbank_id, name, dosage, frequency, route FROM medications WHERE id = ?;", (new_id,))
    new_med = _normalize_medication_row(dict(cursor.fetchone()))

    conn.close()
    return jsonify(new_med)

@app.route('/api/medications/<int:med_id>', methods=['DELETE'])
def remove_medication(med_id):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM medications WHERE status = 'Active';")
    count = cursor.fetchone()[0]
    if count <= 2:
        conn.close()
        return jsonify({"error": "At least 2 active medications required for DDI analysis"}), 400

    cursor.execute("DELETE FROM medications WHERE id = ?;", (med_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "removed_id": med_id})

@app.route('/api/evaluate', methods=['POST'])
def evaluate_regimen():
    data = request.json or {}
    selected_drugs = data.get('selectedDrugs') or data.get('medications') or []
    if not selected_drugs or len(selected_drugs) < 2:
        return jsonify({"error": "At least 2 drugs are required for evaluation"}), 400

    medications = []
    for drug in selected_drugs:
        lookup = _medication_lookup_key(drug)
        medications.append({
            "id": lookup,
            "drugbank_id": drug.get("drugbank_id") or lookup,
            "name": drug.get("name"),
        })

    try:
        orchestrator = get_orchestrator()
        engine_meta = orchestrator.health()
        regimen_result = orchestrator.evaluate_regimen(
            medications=medications,
            patient_id=data.get("patient_id"),
            fold=data.get("fold"),
        )
        payload = orchestrator_regimen_to_portal(
            selected_drugs,
            regimen_result,
            engine_meta=engine_meta,
        )
        return jsonify(payload)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "error": "DDI evaluation failed",
            "detail": str(exc),
        }), 500

@app.route('/api/catalog', methods=['GET'])
def get_catalog():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM drug_catalog;")
    items = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify(items)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5001"))
    print(f"Starting T-GNN Clinical DDI Backend Server on http://localhost:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=True)
