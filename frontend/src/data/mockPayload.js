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

// Known pairwise DDI knowledge graph database
export const knownPairwiseDatabase = {
  "Pantoprazole+Modafinil": {
    probability: 0.91,
    severity: "MAJOR",
    interaction: "YES",
    mechanism: "CYP2C19 Inhibition & CYP3A4 Induction",
    pk: ["metabolism", "elimination", "absorption"],
    pd: ["Altered Serum Bioavailability", "Risk of Gastric Hypersecretion"],
    targets: ["CYP2C19 Isoenzyme", "CYP3A4 Isoenzyme"],
    drugbank1: 26, drugbank2: 5, cpic1: 8, cpic2: 0,
    summary: "Pantoprazole and Modafinil exhibit significant metabolic pathway overlap via CYP2C19 and CYP3A4. Modafinil acts as a CYP2C19 inhibitor, elevating Pantoprazole plasma concentrations."
  },
  "Modafinil+Lisinopril": {
    probability: 0.68,
    severity: "MODERATE",
    interaction: "YES",
    mechanism: "Central Sympathetic Activation vs ACE Antagonism",
    pk: ["elimination"],
    pd: ["Hypertensive Counteraction", "Blood Pressure Fluctuation"],
    targets: ["Angiotensin-Converting Enzyme (ACE)"],
    drugbank1: 5, drugbank2: 18, cpic1: 0, cpic2: 4,
    summary: "Modafinil increases central sympathetic tone, which may partially attenuate the antihypertensive therapeutic effect of Lisinopril."
  },
  "Atorvastatin+Ciprofloxacin": {
    probability: 0.88,
    severity: "MAJOR",
    interaction: "YES",
    mechanism: "CYP3A4 Inhibition leading to Statin Accumulation",
    pk: ["metabolism", "clearance"],
    pd: ["Rhabdomyolysis Risk", "Elevated Creatine Kinase"],
    targets: ["HMG-CoA Reductase", "CYP3A4"],
    drugbank1: 32, drugbank2: 14, cpic1: 6, cpic2: 2,
    summary: "Ciprofloxacin inhibits hepatic CYP3A4, significantly reducing Atorvastatin clearance and heightening risk of severe statin-induced myopathy."
  },
  "Metformin+Ciprofloxacin": {
    probability: 0.72,
    severity: "MODERATE",
    interaction: "YES",
    mechanism: "OCT2 Transporter Renal Competition",
    pk: ["renal excretion"],
    pd: ["Lactic Acidosis Risk", "Altered Renal Clearance"],
    targets: ["Organic Cation Transporter 2 (OCT2)"],
    drugbank1: 15, drugbank2: 14, cpic1: 2, cpic2: 2,
    summary: "Ciprofloxacin competes with Metformin for OCT2 renal tubular secretion, potentially increasing systemic exposure of Metformin."
  },
  "Lisinopril+Metformin": {
    probability: 0.54,
    severity: "MODERATE",
    interaction: "YES",
    mechanism: "Hemodynamic Synergy & Glycemic Regulation",
    pk: ["renal elimination"],
    pd: ["Hypoglycemia Potentiation", "Renal Clearance Shift"],
    targets: ["Renal Tubular Epithelium"],
    drugbank1: 18, drugbank2: 15, cpic1: 4, cpic2: 2,
    summary: "Co-administration of Lisinopril and Metformin may enhance insulin sensitivity and modestly increase risk of mild hypoglycemia."
  },
  "Pantoprazole+Lisinopril": {
    probability: 0.38,
    severity: "LOW",
    interaction: "YES",
    mechanism: "Gastric pH Dependent Bioavailability",
    pk: ["absorption"],
    pd: ["Modest Peak Bioavailability Shift"],
    targets: ["H+/K+-ATPase Pump"],
    drugbank1: 26, drugbank2: 18, cpic1: 8, cpic2: 4,
    summary: "Pantoprazole elevates gastric pH which minimally affects Lisinopril absorption. Clinical significance is generally low."
  },
  "Ciprofloxacin+Amlodipine": {
    probability: 0.76,
    severity: "MODERATE",
    interaction: "YES",
    mechanism: "CYP3A4 Inhibition of Calcium Channel Blocker Clearance",
    pk: ["metabolism"],
    pd: ["Potentiated Hypotension", "Peripheral Edema"],
    targets: ["L-type Calcium Channels"],
    drugbank1: 14, drugbank2: 21, cpic1: 2, cpic2: 3,
    summary: "Ciprofloxacin inhibits metabolism of Amlodipine, increasing serum concentrations and elevating hypotensive risk."
  },
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

/**
 * Dynamic Polypharmacy / Multi-Drug Evaluation Engine
 * Evaluates any number of drugs (2, 3, 4, 5... up to 15+)
 */
export function evaluateMultiDrugRegimen(selectedDrugs) {
  if (!selectedDrugs || selectedDrugs.length < 2) {
    return samplePredictionsDatabase["default"];
  }

  const pairs = [];
  const drugCounts = {};

  // Initialize per-drug evidence counters
  selectedDrugs.forEach(d => {
    const seed = Math.abs(d.name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0));
    drugCounts[d.name] = {
      drugbank: (seed % 25) + 5,
      cpic: (seed % 7),
      twosides: (seed % 3) > 0,
      offsides: true,
    };
  });

  // Calculate all pairwise combinations
  for (let i = 0; i < selectedDrugs.length; i++) {
    for (let j = i + 1; j < selectedDrugs.length; j++) {
      const d1 = selectedDrugs[i];
      const d2 = selectedDrugs[j];
      const pairKey = `${d1.name}+${d2.name}`;
      const revKey = `${d2.name}+${d1.name}`;

      let known = knownPairwiseDatabase[pairKey] || knownPairwiseDatabase[revKey];

      if (!known) {
        // Procedurally generate deterministic risk for unlisted drug pairs
        const hash = (d1.name.charCodeAt(0) * 31 + d2.name.charCodeAt(0) * 17) % 100;
        const prob = Math.round((0.35 + (hash / 200)) * 100) / 100;
        let sev = "LOW";
        if (prob >= 0.8) sev = "MAJOR";
        else if (prob >= 0.55) sev = "MODERATE";

        known = {
          probability: prob,
          severity: sev,
          interaction: prob > 0.4 ? "YES" : "NO",
          mechanism: prob >= 0.7 ? "CYP Enzyme Competition" : "Additive Pharmacodynamic Effect",
          pk: ["metabolism", "elimination"],
          pd: ["Therapeutic Potentiation"],
          targets: ["Hepatic Isoenzymes"],
          summary: `Co-administration of ${d1.name} and ${d2.name} presents potential ${sev.toLowerCase()}-level pharmacokinetic interaction in temporal GNN graph node analysis.`
        };
      }

      pairs.push({
        drug1: d1,
        drug2: d2,
        pairKey: `${d1.name} ↔ ${d2.name}`,
        probability: known.probability,
        severity: known.severity,
        interaction: known.interaction,
        mechanism: known.mechanism,
        pk: known.pk,
        pd: known.pd,
        targets: known.targets,
        summary: known.summary,
      });
    }
  }

  // Sort pairs by probability descending so highest risk pair is first
  pairs.sort((a, b) => b.probability - a.probability);

  const highestPair = pairs[0];
  const maxProbability = highestPair ? highestPair.probability : 0.74;
  
  let overallSeverity = "LOW";
  if (pairs.some(p => p.severity === "MAJOR" || p.severity === "CRITICAL")) {
    overallSeverity = "MAJOR";
  } else if (pairs.some(p => p.severity === "MODERATE")) {
    overallSeverity = "MODERATE";
  }

  const majorCount = pairs.filter(p => p.severity === "MAJOR").length;
  const modCount = pairs.filter(p => p.severity === "MODERATE").length;
  const lowCount = pairs.filter(p => p.severity === "LOW").length;

  // Aggregate PK, PD, Mechanisms & Targets across all pairs
  const allPK = Array.from(new Set(pairs.flatMap(p => p.pk || [])));
  const allPD = Array.from(new Set(pairs.flatMap(p => p.pd || [])));
  const allMechs = Array.from(new Set(pairs.flatMap(p => p.mechanism || [])));
  const allTargets = Array.from(new Set(pairs.flatMap(p => p.targets || [])));

  const drugNamesStr = selectedDrugs.map(d => d.name).join(", ");
  const ragSummary = `Evaluation of ${selectedDrugs.length}-drug polypharmacy regimen (${drugNamesStr}): Analyzed ${pairs.length} distinct pairwise interaction channels in the T-GNN graph network. Highest risk interaction observed between ${highestPair.drug1.name} and ${highestPair.drug2.name} (${Math.round(highestPair.probability * 100)}% probability, ${highestPair.severity} severity). Overlaps detected across ${allPK.join(", ")} pharmacokinetic pathways and ${allMechs.slice(0, 2).join(", ")} mechanisms.`;

  return {
    selectedDrugs,
    pairs,
    highestPair,
    pairCount: pairs.length,
    majorCount,
    modCount,
    lowCount,
    prediction: {
      interaction: "YES",
      probability: maxProbability,
      severity: overallSeverity,
      source: `T-GNN Clinical Engine v2.4 (${selectedDrugs.length}-Drug Graph)`,
      alertCode: `CDSS-DDI-POLY-${selectedDrugs.length}D-${Math.round(maxProbability * 100)}`,
    },
    rag_explanation: {
      summary: ragSummary,
      shared_targets: allTargets,
      shared_mechanisms: allMechs,
      shared_pk: allPK,
      shared_pd_effects: allPD,
    },
    evidence: {
      drugCounts,
      drugbank: { total: Object.values(drugCounts).reduce((acc, c) => acc + c.drugbank, 0) },
      cpic: { total: Object.values(drugCounts).reduce((acc, c) => acc + c.cpic, 0) },
      twosides: { observed: majorCount > 0 || modCount > 0 },
      offsides: { available: true }
    }
  };
}
