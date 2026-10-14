import React, { useState } from 'react';
import './DrugPairCard.css';
import { PRESET_DRUG_PAIRS } from '../data/commonMedications';
import { BoltLabelIcon, PillLabelIcon } from './SegmentIcons';

function medKey(med) {
  return med.id || med.drugbank_id || med.name;
}

function uniquePairCount(n) {
  if (n < 2) return 0;
  return (n * (n - 1)) / 2;
}

export function DrugPairCard({
  selectedDrugs = [],
  availableMeds = [],
  evaluationData,
  onRunEvaluation,
  isEvaluating,
  onSelectDrug1,
  onSelectDrug2,
  onSelectPresetPair,
  onToggleEvalMedication,
  onSelectAllEval,
}) {
  const [matrixOpen, setMatrixOpen] = useState(false);
  const pairs = evaluationData?.pairs || [];
  const isMultiDrug = selectedDrugs.length >= 3;
  const canEvaluate = selectedDrugs.length >= 2 && !isEvaluating;
  const expectedPairs = uniquePairCount(selectedDrugs.length);

  const allEvalSelected =
    availableMeds.length > 0 &&
    availableMeds.every((med) => selectedDrugs.some((m) => medKey(m) === medKey(med)));

  const drug1 = selectedDrugs[0] || availableMeds[0] || {};
  const drug2 = selectedDrugs[1] || availableMeds[1] || {};

  return (
    <div className="screenshot-card appointments-card">
      <section className="eval-med-picker-section" aria-labelledby="eval-picker-heading">
        <div className="eval-picker-header">
          <div>
            <h3 className="card-main-title" id="eval-picker-heading">
              Medications for this evaluation
            </h3>
            <p className="section-helper inline-helper">
              Choose from the patient&apos;s active orders. This does not change the medication list.
            </p>
          </div>
          {onSelectAllEval && availableMeds.length > 0 && (
            <button type="button" className="select-all-btn eval-select-all-btn" onClick={onSelectAllEval}>
              {allEvalSelected ? 'Clear evaluation selection' : `Select all (${availableMeds.length})`}
            </button>
          )}
        </div>

        <div className="eval-selection-stats" role="status">
          <span>
            <strong>{selectedDrugs.length}</strong> medication{selectedDrugs.length === 1 ? '' : 's'} selected
          </span>
          <span className="eval-stat-sep" aria-hidden="true">
            ·
          </span>
          <span>
            <strong>{expectedPairs}</strong> unique pair{expectedPairs === 1 ? '' : 's'} to evaluate
          </span>
        </div>

        {selectedDrugs.length < 2 && (
          <p className="eval-validation-msg" role="alert">
            Select at least two medications to run an interaction evaluation.
          </p>
        )}

        <div className="mar-table-wrapper-screenshot eval-picker-table-wrap">
          <table className="screenshot-mar-table eval-picker-table">
            <thead>
              <tr>
                <th>Include</th>
                <th>DrugBank ID</th>
                <th>Medication</th>
                <th>Dosage</th>
                <th>Frequency</th>
              </tr>
            </thead>
            <tbody>
              {availableMeds.map((med) => {
                const isSelected = selectedDrugs.some((m) => medKey(m) === medKey(med));
                return (
                  <tr key={medKey(med)} className={isSelected ? 'selected-pair-row' : ''}>
                    <td className="center-col">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        aria-label={`Include ${med.name} in evaluation`}
                        onChange={() => onToggleEvalMedication && onToggleEvalMedication(med)}
                      />
                    </td>
                    <td className="mono-code">{med.id || med.drugbank_id}</td>
                    <td className="drug-name-col">
                      <strong>{med.name}</strong>
                    </td>
                    <td>{med.dosage || '—'}</td>
                    <td>{med.frequency || '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <div className="card-top-title-row">
        <div className="title-group">
          <h3 className="card-main-title">
            {isMultiDrug ? 'Evaluation regimen' : 'Primary drug pair focus'}
          </h3>
          <span className="selected-count-badge">
            {selectedDrugs.length} in regimen
            {pairs.length > 0 ? ` · ${pairs.length} result pair${pairs.length === 1 ? '' : 's'}` : ''}
          </span>
          <span className="section-helper inline-helper">
            Optional: set Drug A and B for pair-focused controls and presets.
          </span>
        </div>

        {onRunEvaluation && (
          <button
            type="button"
            className={`run-eval-primary-btn ${isEvaluating ? 'evaluating' : ''}`}
            onClick={onRunEvaluation}
            disabled={!canEvaluate}
            aria-busy={isEvaluating}
          >
            {isEvaluating ? (
              <>
                <span className="eval-spinner" aria-hidden="true" />
                <span>Evaluating interactions…</span>
              </>
            ) : (
              <span>Run interaction evaluation</span>
            )}
          </button>
        )}
      </div>

      <div className="drug-dropdown-toolbar">
        <div className="dropdown-tool-group">
          <label className="dropdown-label font-bold" htmlFor="drug-a-select">
            <PillLabelIcon /> Drug A
          </label>
          <select
            id="drug-a-select"
            className="med-select-dropdown"
            value={drug1.id || drug1.drugbank_id || ''}
            onChange={(e) => onSelectDrug1 && onSelectDrug1(e.target.value)}
          >
            {availableMeds.map((med) => (
              <option key={med.id || med.drugbank_id} value={med.id || med.drugbank_id}>
                {med.name} ({med.id || med.drugbank_id})
              </option>
            ))}
          </select>
        </div>

        <span className="dropdown-vs-badge" aria-hidden="true">
          vs
        </span>

        <div className="dropdown-tool-group">
          <label className="dropdown-label font-bold" htmlFor="drug-b-select">
            <PillLabelIcon /> Drug B
          </label>
          <select
            id="drug-b-select"
            className="med-select-dropdown"
            value={drug2.id || drug2.drugbank_id || ''}
            onChange={(e) => onSelectDrug2 && onSelectDrug2(e.target.value)}
          >
            {availableMeds.map((med) => (
              <option key={med.id || med.drugbank_id} value={med.id || med.drugbank_id}>
                {med.name} ({med.id || med.drugbank_id})
              </option>
            ))}
          </select>
        </div>

        {onSelectPresetPair && (
          <div className="dropdown-tool-group preset-group">
            <label className="dropdown-label font-bold" htmlFor="preset-pair-select">
              <BoltLabelIcon /> Common pair preset
            </label>
            <select
              id="preset-pair-select"
              className="med-select-dropdown preset-dropdown"
              defaultValue=""
              onChange={(e) => e.target.value && onSelectPresetPair(e.target.value)}
            >
              <option value="" disabled>
                Choose a preset pair
              </option>
              {PRESET_DRUG_PAIRS.map((preset, idx) => (
                <option key={idx} value={preset.pair.join('+')}>
                  {preset.label}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className="selected-meds-container">
        <div className="sub-section-label">Selected for evaluation</div>

        {selectedDrugs.length === 0 ? (
          <p className="section-helper">No medications selected yet.</p>
        ) : (
          <div className={`drug-items-grid ${selectedDrugs.length > 3 ? 'compact-grid' : ''}`}>
            {selectedDrugs.map((drug, index) => (
              <div key={medKey(drug) || index} className="drug-appointment-item accent-highlight">
                <div className="highlight-accent-line accent" />
                <div className="item-content-group">
                  <div className="drug-header-title">
                    <strong className="drug-name-text">{drug.name}</strong>
                    <span className="code-chip accent">{drug.id || drug.drugbank_id}</span>
                  </div>
                  <span className="item-subtext">
                    #{index + 1} · {drug.dosage || '—'} · {drug.frequency || '—'}
                  </span>
                </div>
                <div className="right-checks">
                  {onToggleEvalMedication && (
                    <button
                      type="button"
                      className="eval-remove-chip-btn"
                      onClick={() => onToggleEvalMedication(drug)}
                      aria-label={`Remove ${drug.name} from evaluation`}
                    >
                      Remove
                    </button>
                  )}
                  <span className="drug-num-badge">{index + 1}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {pairs.length > 0 && (
        <div className="pairwise-matrix-section">
          <div className="matrix-header-row">
            <button
              type="button"
              className="matrix-toggle-btn"
              onClick={() => setMatrixOpen((o) => !o)}
              aria-expanded={matrixOpen}
            >
              <h4 className="matrix-title">
                Pairwise breakdown ({pairs.length} pairs)
              </h4>
              <svg
                className={`chevron-toggle ${matrixOpen ? 'open' : ''}`}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                aria-hidden="true"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <div className="matrix-summary-pills">
              {evaluationData?.majorCount > 0 && (
                <span className="matrix-pill major">{evaluationData.majorCount} major</span>
              )}
              {evaluationData?.modCount > 0 && (
                <span className="matrix-pill moderate">{evaluationData.modCount} moderate</span>
              )}
              {evaluationData?.lowCount > 0 && (
                <span className="matrix-pill low">{evaluationData.lowCount} low</span>
              )}
            </div>
          </div>

          {matrixOpen && (
            <div className="pairwise-cards-list">
              {pairs.map((pair, idx) => (
                <div key={idx} className={`pair-breakdown-card severity-${pair.severity.toLowerCase()}`}>
                  <div className="pair-card-top">
                    <div className="pair-drugs-names">
                      <span className="pair-drug-badge">{pair.drug1.name}</span>
                      <span className="pair-arrow" aria-hidden="true">
                        ↔
                      </span>
                      <span className="pair-drug-badge">{pair.drug2.name}</span>
                    </div>
                    <div className="pair-metrics">
                      <span className={`pair-sev-tag ${pair.severity.toLowerCase()}`}>{pair.severity}</span>
                      <span className="pair-prob-val">{Math.round(pair.probability * 100)}% model score</span>
                    </div>
                  </div>

                  {pair.mechanism && (
                    <div className="pair-card-mechanism">
                      <span className="mech-label">Mechanism:</span> {pair.mechanism}
                    </div>
                  )}
                  {pair.summary && <p className="pair-card-summary">{pair.summary}</p>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
