import { calculateAgeFromDob } from '../components/dobDateUtils';

const DB_ID_PATTERN = /^DB\d+$/i;

export function looksLikeDrugbankId(value) {
  return DB_ID_PATTERN.test(String(value || '').trim());
}

export function buildDrugLookup(...medLists) {
  const lookup = new Map();
  medLists.flat().filter(Boolean).forEach((med) => {
    const dbId = String(med.id || med.drugbank_id || '').trim().toUpperCase();
    if (!dbId) return;
    const existing = lookup.get(dbId);
    const name = (med.name || '').trim();
    if (!existing || (name && !looksLikeDrugbankId(name))) {
      lookup.set(dbId, { id: dbId, name });
    }
  });
  return lookup;
}

export function resolveDrugDisplay(drug, lookup) {
  const dbId = String(drug?.id || drug?.drugbank_id || '').trim().toUpperCase();
  let name = (drug?.name || '').trim();
  if (looksLikeDrugbankId(name)) name = '';

  const fromLookup = dbId ? lookup.get(dbId) : null;
  if (fromLookup?.name && !looksLikeDrugbankId(fromLookup.name)) {
    name = fromLookup.name.trim();
  }

  let nameUnresolved = false;
  if (!name) {
    name = 'Name not resolved';
    nameUnresolved = true;
  }

  return {
    id: dbId || null,
    name,
    nameUnresolved,
  };
}

export function pairIdentityKey(drug1, drug2) {
  const a = String(drug1?.id || '').toUpperCase();
  const b = String(drug2?.id || '').toUpperCase();
  return [a, b].sort().join('|');
}

export function expectedPairCount(medCount) {
  return medCount >= 2 ? (medCount * (medCount - 1)) / 2 : 0;
}

export function normalizePatientForReport(patient) {
  if (!patient) return patient;
  const normalized = { ...patient };
  if (normalized.dob) {
    const derived = calculateAgeFromDob(normalized.dob);
    if (derived != null) {
      normalized.age = derived;
    }
  }
  return normalized;
}

/**
 * Align pair drug labels with MAR / evaluation selection; dedupe unordered pairs.
 */
export function prepareEvaluationForReport(evaluation, patient) {
  if (!evaluation) return null;

  const lookup = buildDrugLookup(
    evaluation.selectedDrugs || [],
    patient?.medications || [],
  );

  const seen = new Set();
  const pairs = (evaluation.pairs || [])
    .map((pair) => {
      const drug1 = resolveDrugDisplay(pair.drug1 || {}, lookup);
      const drug2 = resolveDrugDisplay(pair.drug2 || {}, lookup);
      return {
        ...pair,
        drug1,
        drug2,
        pairKey: `${drug1.name} ↔ ${drug2.name}`,
      };
    })
    .filter((pair) => {
      const key = pairIdentityKey(pair.drug1, pair.drug2);
      if (!key || key === '|') return false;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

  const medCount = (evaluation.selectedDrugs || []).length;
  const expected = evaluation.expectedPairCount ?? evaluation.evaluation_meta?.expected_pair_count ?? expectedPairCount(medCount);

  return {
    ...evaluation,
    pairs,
    pairCount: pairs.length,
    expectedPairCount: expected,
    countMismatch: expected > 0 && pairs.length !== expected,
  };
}
