export const initialPatient = {
  mrn: "MRN-9842-7019",
  name: "Eleanor Vance",
  firstName: "Eleanor",
  lastName: "Vance",
  initials: "EV",
  age: 72,
  dob: "1954-07-22",
  gender: "Female",
  language: "English",
  phone: "(406) 555-0120",
  emergencyContact: "(480) 555-0103",
  email: "eleanor.vance@example.com",
  address: "2972 Westheimer Rd. Santa Ana, Illinois 85486",
  weight: "68 kg",
  room: "Bed 304-B (Step-down Unit)",
  physician: "Dr. Jordan Hughes",
  medications: [
    { id: "DB00213", name: "Pantoprazole", dosage: "40 mg", frequency: "QD (Daily)", route: "Oral" },
    { id: "DB00745", name: "Modafinil", dosage: "200 mg", frequency: "QAM (Morning)", route: "Oral" },
    { id: "DB00722", name: "Lisinopril", dosage: "10 mg", frequency: "QD (Daily)", route: "Oral" },
    { id: "DB00331", name: "Metformin", dosage: "500 mg", frequency: "BID (Twice Daily)", route: "Oral" },
    { id: "DB01076", name: "Atorvastatin", dosage: "20 mg", frequency: "QHS (Bedtime)", route: "Oral" },
    { id: "DB00537", name: "Ciprofloxacin", dosage: "250 mg", frequency: "BID", route: "Oral" },
    { id: "DB00381", name: "Amlodipine", dosage: "5 mg", frequency: "QD", route: "Oral" },
  ],
};

export const samplePredictionsDatabase = {
  "Pantoprazole+Modafinil": {
    prediction: {
      interaction: "YES",
      probability: 0.91,
      severity: "MAJOR",
      source: "T-GNN Clinical Engine v2.4",
      alertCode: "CDSS-DDI-CRITICAL-091",
    },
    rag_explanation: {
      summary:
        "Pantoprazole and Modafinil exhibit significant metabolic pathway overlap in hepatic biotransformation. Both agents interact via Cytochrome P450 2C19 (CYP2C19) and 3A4 (CYP3A4) pathways. Modafinil acts as a potent CYP2C19 inhibitor and CYP3A4 inducer, which significantly alters Pantoprazole clearance and systemic bioavailability. Co-administration elevates risk of altered serum levels and gastrointestinal / neurological adverse drug reactions (ADRs).",
      shared_targets: [],
      shared_mechanisms: ["Hepatic CYP2C19 Inhibition", "CYP3A4 Enzyme Induction"],
      shared_pk: ["absorption", "elimination", "metabolism"],
      shared_pd_effects: ["Altered Serum Bioavailability", "Risk of Gastric Hypersecretion"],
    },
    evidence: {
      drugbank: { drug_1_count: 26, drug_2_count: 5 },
      cpic: { drug_1_count: 8, drug_2_count: 0 },
      twosides: { observed: false, ddi_type_ids: [], ddi_type_count: 0 },
      offsides: { drug_1_available: true, drug_2_available: true },
    },
  },
  "default": {
    prediction: {
      interaction: "YES",
      probability: 0.74,
      severity: "MODERATE",
      source: "T-GNN Clinical Engine v2.4",
      alertCode: "CDSS-DDI-MOD-074",
    },
    rag_explanation: {
      summary:
        "Co-administration of selected agents presents potential pharmacokinetic interaction via competitive enzyme binding. Close monitoring of therapeutic response and renal clearance is recommended.",
      shared_targets: ["CYP3A4 Isoenzyme"],
      shared_mechanisms: ["Substrate Competition"],
      shared_pk: ["metabolism", "elimination"],
      shared_pd_effects: ["Additive Hypotensive Effect"],
    },
    evidence: {
      drugbank: { drug_1_count: 12, drug_2_count: 9 },
      cpic: { drug_1_count: 3, drug_2_count: 2 },
      twosides: { observed: true, ddi_type_ids: ["DDI_104"], ddi_type_count: 1 },
      offsides: { drug_1_available: true, drug_2_available: true },
    },
  },
};
