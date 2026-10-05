import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clinical_ddi.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Patients Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mrn TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        first_name TEXT,
        last_name TEXT,
        initials TEXT,
        age INTEGER,
        dob TEXT,
        gender TEXT,
        language TEXT,
        phone TEXT,
        emergency_contact TEXT,
        email TEXT,
        address TEXT,
        weight TEXT,
        room TEXT,
        physician TEXT
    );
    """)

    # 2. Medications Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS medications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        drugbank_id TEXT NOT NULL,
        name TEXT NOT NULL,
        dosage TEXT,
        frequency TEXT,
        route TEXT,
        status TEXT DEFAULT 'Active',
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    );
    """)

    # 3. Drug Catalog Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS drug_catalog (
        drugbank_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        category TEXT
    );
    """)

    # 4. Pairwise DDI Knowledge Base Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pairwise_ddi_knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        drug1_name TEXT NOT NULL,
        drug2_name TEXT NOT NULL,
        probability REAL NOT NULL,
        severity TEXT NOT NULL,
        interaction TEXT DEFAULT 'YES',
        mechanism TEXT,
        pk_pathways TEXT,
        pd_effects TEXT,
        targets TEXT,
        summary TEXT,
        drugbank_d1_count INTEGER DEFAULT 12,
        drugbank_d2_count INTEGER DEFAULT 8,
        cpic_d1_count INTEGER DEFAULT 4,
        cpic_d2_count INTEGER DEFAULT 2
    );
    """)

    # 5. RAG Knowledge Base Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rag_knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pathway_name TEXT NOT NULL,
        property_type TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        clinical_effect TEXT NOT NULL
    );
    """)

    # Seed Initial Data if empty
    cursor.execute("SELECT COUNT(*) FROM patients;")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
        INSERT INTO patients (
            mrn, name, first_name, last_name, initials, age, dob, gender,
            language, phone, emergency_contact, email, address, weight, room, physician
        ) VALUES (
            'MRN-9842-7019', 'Eleanor Vance', 'Eleanor', 'Vance', 'EV', 72, '1954-07-22', 'Female',
            'English', '(406) 555-0120', '(480) 555-0103', 'eleanor.vance@example.com',
            '2972 Westheimer Rd. Santa Ana, Illinois 85486', '68 kg', 'Bed 304-B (Step-down Unit)', 'Dr. Jordan Hughes'
        );
        """)
        patient_id = cursor.lastrowid

        # Seed Initial Medications
        initial_meds = [
            ("DB00213", "Pantoprazole", "40 mg", "QD (Daily)", "Oral"),
            ("DB00745", "Modafinil", "200 mg", "QAM (Morning)", "Oral"),
            ("DB00722", "Lisinopril", "10 mg", "QD (Daily)", "Oral"),
            ("DB00331", "Metformin", "500 mg", "BID (Twice Daily)", "Oral"),
            ("DB01076", "Atorvastatin", "20 mg", "QHS (Bedtime)", "Oral"),
            ("DB00537", "Ciprofloxacin", "250 mg", "BID", "Oral"),
            ("DB00381", "Amlodipine", "5 mg", "QD", "Oral"),
        ]
        for db_id, name, dosage, freq, route in initial_meds:
            cursor.execute("""
            INSERT INTO medications (patient_id, drugbank_id, name, dosage, frequency, route)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (patient_id, db_id, name, dosage, freq, route))

    # Seed Drug Catalog
    cursor.execute("SELECT COUNT(*) FROM drug_catalog;")
    if cursor.fetchone()[0] == 0:
        catalog = [
            ("DB00213", "Pantoprazole", "Proton Pump Inhibitor", "Gastrointestinal"),
            ("DB00745", "Modafinil", "Central Nervous System Stimulant", "Neurological"),
            ("DB00722", "Lisinopril", "Angiotensin-Converting Enzyme (ACE) Inhibitor", "Cardiovascular"),
            ("DB00331", "Metformin", "Biguanide Antidiabetic Agent", "Endocrine"),
            ("DB01076", "Atorvastatin", "HMG-CoA Reductase Inhibitor (Statin)", "Lipid-lowering"),
            ("DB00537", "Ciprofloxacin", "Fluoroquinolone Antibiotic", "Anti-infective"),
            ("DB00381", "Amlodipine", "Dihydropyridine Calcium Channel Blocker", "Cardiovascular"),
            ("DB01098", "Rosuvastatin", "HMG-CoA Reductase Inhibitor", "Lipid-lowering"),
            ("DB00682", "Warfarin", "Vitamin K Antagonist Anticoagulant", "Hematologic"),
        ]
        cursor.executemany("INSERT INTO drug_catalog VALUES (?, ?, ?, ?);", catalog)

    # Seed Pairwise DDI Knowledge Base
    cursor.execute("SELECT COUNT(*) FROM pairwise_ddi_knowledge;")
    if cursor.fetchone()[0] == 0:
        ddi_pairs = [
            ("Pantoprazole", "Modafinil", 0.91, "MAJOR", "YES", 
             "CYP2C19 Inhibition & CYP3A4 Induction",
             json.dumps(["metabolism", "elimination", "absorption"]),
             json.dumps(["Altered Serum Bioavailability", "Risk of Gastric Hypersecretion"]),
             json.dumps(["CYP2C19 Isoenzyme", "CYP3A4 Isoenzyme"]),
             "Pantoprazole and Modafinil exhibit significant metabolic pathway overlap via CYP2C19 and CYP3A4. Modafinil acts as a CYP2C19 inhibitor, elevating Pantoprazole plasma concentrations.",
             26, 5, 8, 0),
            ("Modafinil", "Lisinopril", 0.68, "MODERATE", "YES",
             "Central Sympathetic Activation vs ACE Antagonism",
             json.dumps(["elimination"]),
             json.dumps(["Hypertensive Counteraction", "Blood Pressure Fluctuation"]),
             json.dumps(["Angiotensin-Converting Enzyme (ACE)"]),
             "Modafinil increases central sympathetic tone, which may partially attenuate the antihypertensive therapeutic effect of Lisinopril.",
             5, 18, 0, 4),
            ("Atorvastatin", "Ciprofloxacin", 0.88, "MAJOR", "YES",
             "CYP3A4 Inhibition leading to Statin Accumulation",
             json.dumps(["metabolism", "clearance"]),
             json.dumps(["Rhabdomyolysis Risk", "Elevated Creatine Kinase"]),
             json.dumps(["HMG-CoA Reductase", "CYP3A4"]),
             "Ciprofloxacin inhibits hepatic CYP3A4, significantly reducing Atorvastatin clearance and heightening risk of severe statin-induced myopathy.",
             32, 14, 6, 2),
            ("Metformin", "Ciprofloxacin", 0.72, "MODERATE", "YES",
             "OCT2 Transporter Renal Competition",
             json.dumps(["renal excretion"]),
             json.dumps(["Lactic Acidosis Risk", "Altered Renal Clearance"]),
             json.dumps(["Organic Cation Transporter 2 (OCT2)"]),
             "Ciprofloxacin competes with Metformin for OCT2 renal tubular secretion, potentially increasing systemic exposure of Metformin.",
             15, 14, 2, 2),
            ("Lisinopril", "Metformin", 0.54, "MODERATE", "YES",
             "Hemodynamic Synergy & Glycemic Regulation",
             json.dumps(["renal elimination"]),
             json.dumps(["Hypoglycemia Potentiation", "Renal Clearance Shift"]),
             json.dumps(["Renal Tubular Epithelium"]),
             "Co-administration of Lisinopril and Metformin may enhance insulin sensitivity and modestly increase risk of mild hypoglycemia.",
             18, 15, 4, 2),
            ("Pantoprazole", "Lisinopril", 0.38, "LOW", "YES",
             "Gastric pH Dependent Bioavailability",
             json.dumps(["absorption"]),
             json.dumps(["Modest Peak Bioavailability Shift"]),
             json.dumps(["H+/K+-ATPase Pump"]),
             "Pantoprazole elevates gastric pH which minimally affects Lisinopril absorption. Clinical significance is generally low.",
             26, 18, 8, 4),
            ("Ciprofloxacin", "Amlodipine", 0.76, "MODERATE", "YES",
             "CYP3A4 Inhibition of Calcium Channel Blocker Clearance",
             json.dumps(["metabolism"]),
             json.dumps(["Potentiated Hypotension", "Peripheral Edema"]),
             json.dumps(["L-type Calcium Channels"]),
             "Ciprofloxacin inhibits metabolism of Amlodipine, increasing serum concentrations and elevating hypotensive risk.",
             14, 21, 2, 3)
        ]
        cursor.executemany("""
        INSERT INTO pairwise_ddi_knowledge (
            drug1_name, drug2_name, probability, severity, interaction, mechanism,
            pk_pathways, pd_effects, targets, summary, drugbank_d1_count, drugbank_d2_count,
            cpic_d1_count, cpic_d2_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, ddi_pairs)

    # Seed RAG Knowledge Base
    cursor.execute("SELECT COUNT(*) FROM rag_knowledge_base;")
    if cursor.fetchone()[0] == 0:
        rag_items = [
            ("CYP2C19 & CYP3A4 Pathway", "PK", "Hepatic CYP Metabolism Overlap", "Altered Drug Bioavailability"),
            ("CYP2C19 & CYP3A4 Pathway", "MECHANISM", "Hepatic CYP2C19 Inhibition", "Enzyme Inhibition Risk"),
            ("CYP2C19 & CYP3A4 Pathway", "MECHANISM", "CYP3A4 Enzyme Induction", "Altered Clearance Speed"),
            ("CYP2C19 & CYP3A4 Pathway", "TARGET", "Cytochrome P450 Isoenzymes", "Receptor & Enzyme Competition"),
            ("CYP2C19 & CYP3A4 Pathway", "PD EFFECT", "Altered Serum Bioavailability", "Adverse Event Potential"),
            ("CYP2C19 & CYP3A4 Pathway", "PD EFFECT", "Risk of Gastric Hypersecretion", "Adverse Event Potential")
        ]
        cursor.executemany("INSERT INTO rag_knowledge_base (pathway_name, property_type, feature_name, clinical_effect) VALUES (?, ?, ?, ?);", rag_items)

    conn.commit()
    conn.close()
    print(f"Database successfully initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()
