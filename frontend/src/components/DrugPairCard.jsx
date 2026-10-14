import React from 'react';
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
  onRunEvaluation,
  isEvaluating,
  onSelectDrug1,
  onSelectDrug2,
  onSelectPresetPair,
  onToggleEvalMedication,
  onSelectAllEval,
  resultsStale = false,
  evaluationData,
}) {
  const canEvaluate = selectedDrugs.length >= 2 && !isEvaluating;
  const expectedPairs = uniquePairCount(selectedDrugs.length);
  const pairFocusReady = selectedDrugs.length >= 2;

  const allEvalSelected =
    availableMeds.length > 0 &&
    availableMeds.every((med) => selectedDrugs.some((m) => medKey(m) === medKey(med)));

  const drug1 = selectedDrugs[0] || {};
  const drug2 =
    selectedDrugs.find((m, i) => i > 0 && medKey(m) !== medKey(drug1)) ||
    selectedDrugs[1] ||
    {};

  const drug1Value = drug1.id || drug1.drugbank_id || '';
  const drug2Value = drug2.id || drug2.drugbank_id || '';

  return (
    <div className="screenshot-card appointments-card">
      <section className="eval-med-picker-section" aria-labelledby="eval-picker-heading">
        <div className="eval-picker-header">
          <div className="eval-picker-title-row">
            <h3 className="card-main-title" id="eval-picker-heading">
              Select medications
            </h3>
            {selectedDrugs.length > 0 && (
              <span className="eval-count-badge">{selectedDrugs.length} selected</span>
            )}
          </div>
          {onSelectAllEval && availableMeds.length > 0 && (
            <button type="button" className="select-all-btn eval-select-all-btn" onClick={onSelectAllEval}>
              {allEvalSelected ? 'Clear evaluation selection' : `Select all (${availableMeds.length})`}
            </button>
          )}
        </div>

        <div className="eval-selection-stats" role="status">
          <span>
            <strong>{expectedPairs}</strong> unique pair{expectedPairs === 1 ? '' : 's'} to evaluate
          </span>
        </div>

        {selectedDrugs.length < 2 && (
          <p className="eval-validation-msg" role="alert">
            Select at least two medications to run an interaction evaluation.
          </p>
        )}

        <div className="eval-med-box-grid" role="group" aria-labelledby="eval-picker-heading">
          {availableMeds.map((med) => {
            const isSelected = selectedDrugs.some((m) => medKey(m) === medKey(med));
            const inputId = `eval-med-${medKey(med)}`;
            return (
              <div
                key={medKey(med)}
                className={`eval-med-box ${isSelected ? 'is-selected' : ''}`}
              >
                <label className="eval-med-box-label" htmlFor={inputId}>
                  <input
                    id={inputId}
                    type="checkbox"
                    className="eval-med-box-checkbox"
                    checked={isSelected}
                    onChange={() => onToggleEvalMedication && onToggleEvalMedication(med)}
                  />
                  <span className="eval-med-box-name">{med.name}</span>
                </label>
              </div>
            );
          })}
        </div>
      </section>

      {resultsStale && evaluationData && (
        <p className="eval-stale-notice" role="status">
          Results below are from a previous selection. Run evaluation again to update them for the
          current drugs.
        </p>
      )}

      <div className="card-top-title-row eval-actions-row">
        <div className="title-group">
          <span className="section-helper inline-helper">
            Optional: choose Drug A and Drug B from your selection for pair-focused presets.
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

      <div className={`drug-dropdown-toolbar ${!pairFocusReady ? 'is-disabled' : ''}`}>
        <div className="dropdown-tool-group">
          <label className="dropdown-label font-bold" htmlFor="drug-a-select">
            <PillLabelIcon /> Drug A
          </label>
          <select
            id="drug-a-select"
            className="med-select-dropdown"
            value={pairFocusReady ? drug1Value : ''}
            disabled={!pairFocusReady}
            onChange={(e) => onSelectDrug1 && onSelectDrug1(e.target.value)}
          >
            {selectedDrugs.map((med) => (
              <option key={medKey(med)} value={med.id || med.drugbank_id}>
                {med.name}
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
            value={pairFocusReady ? drug2Value : ''}
            disabled={!pairFocusReady}
            onChange={(e) => onSelectDrug2 && onSelectDrug2(e.target.value)}
          >
            {selectedDrugs
              .filter((med) => medKey(med) !== medKey(drug1))
              .map((med) => (
                <option key={medKey(med)} value={med.id || med.drugbank_id}>
                  {med.name}
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
              disabled={!pairFocusReady}
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
    </div>
  );
}
