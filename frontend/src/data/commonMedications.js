export const COMMON_MEDICATIONS = [
  { id: "DB00213", name: "Pantoprazole", dosage: "40 mg", frequency: "QD (Daily)", route: "Oral", category: "Proton Pump Inhibitor" },
  { id: "DB00745", name: "Modafinil", dosage: "200 mg", frequency: "QAM (Morning)", route: "Oral", category: "CNS Stimulant / Wakefulness" },
  { id: "DB00722", name: "Lisinopril", dosage: "10 mg", frequency: "QD (Daily)", route: "Oral", category: "ACE Inhibitor / Antihypertensive" },
  { id: "DB00331", name: "Metformin", dosage: "500 mg", frequency: "BID (Twice Daily)", route: "Oral", category: "Antidiabetic / Biguanide" },
  { id: "DB01076", name: "Atorvastatin", dosage: "20 mg", frequency: "QHS (Bedtime)", route: "Oral", category: "HMG-CoA Reductase Inhibitor" },
  { id: "DB00537", name: "Ciprofloxacin", dosage: "250 mg", frequency: "BID", route: "Oral", category: "Fluoroquinolone Antibiotic" },
  { id: "DB00381", name: "Amlodipine", dosage: "5 mg", frequency: "QD", route: "Oral", category: "Calcium Channel Blocker" },
  { id: "DB00945", name: "Aspirin", dosage: "81 mg", frequency: "QD", route: "Oral", category: "Antiplatelet / NSAID" },
  { id: "DB00682", name: "Warfarin", dosage: "5 mg", frequency: "QD", route: "Oral", category: "Anticoagulant" },
  { id: "DB00338", name: "Omeprazole", dosage: "20 mg", frequency: "QD", route: "Oral", category: "Proton Pump Inhibitor" },
  { id: "DB00641", name: "Simvastatin", dosage: "20 mg", frequency: "QHS", route: "Oral", category: "Statin / Lipid Lowering" },
  { id: "DB00281", name: "Levothyroxine", dosage: "50 mcg", frequency: "QAM", route: "Oral", category: "Thyroid Hormone Replacement" },
  { id: "DB00758", name: "Clopidogrel", dosage: "75 mg", frequency: "QD", route: "Oral", category: "P2Y12 Antiplatelet" },
  { id: "DB00316", name: "Acetaminophen", dosage: "500 mg", frequency: "Q6H PRN", route: "Oral", category: "Analgesic / Antipyretic" },
  { id: "DB01050", name: "Ibuprofen", dosage: "400 mg", frequency: "Q8H PRN", route: "Oral", category: "NSAID" },
];

export const PRESET_DRUG_PAIRS = [
  { label: "Pantoprazole + Modafinil (Hepatic CYP2C19 Risk)", pair: ["DB00213", "DB00745"] },
  { label: "Lisinopril + Amlodipine (Antihypertensive Synergy)", pair: ["DB00722", "DB00381"] },
  { label: "Metformin + Atorvastatin (Cardiometabolic)", pair: ["DB00331", "DB01076"] },
  { label: "Ciprofloxacin + Pantoprazole (Absorption Interaction)", pair: ["DB00537", "DB00213"] },
  { label: "Aspirin + Warfarin (Critical Bleeding Risk)", pair: ["DB00945", "DB00682"] },
  { label: "Clopidogrel + Omeprazole (CYP2C19 Antiplatelet Inhibition)", pair: ["DB00758", "DB00338"] },
];
