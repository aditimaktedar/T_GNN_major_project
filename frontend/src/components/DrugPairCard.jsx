import React from 'react';
import './DrugPairCard.css';
import { PRESET_DRUG_PAIRS } from '../data/commonMedications';

const ACCENT_COLORS = [
  'blue', 'pink', 'purple', 'amber', 'emerald', 'teal'
];

export function DrugPairCard({ 
  selectedDrugs = [], 
  availableMeds = [], 
  evaluationData, 
  onRunEvaluation, 
  isEvaluating,
  onSelectDrug1,
  onSelectDrug2,
  onSelectPresetPair
}) {
  const pairs = evaluationData?.pairs || [];
  const isMultiDrug = selectedDrugs.length >= 3;

  const drug1 = selectedDrugs[0] || availableMeds[0] || {};
  const drug2 = selectedDrugs[1] || availableMeds[1] || {};

  return (
    <div className="screenshot-card appointments-card">
      {/* Top Header Row */}
      <div className="card-top-title-row">
        <div className="title-group">
          <h3 className="card-main-title">
            {isMultiDrug ? 'Evaluated Regimen & Pairwise Matrix' : 'Evaluated Drug Pair'}
          </h3>
          <span className="selected-count-badge">
            {selectedDrugs.length} Drugs Selected {pairs.length > 0 ? `• ${pairs.length} Pair Interactions` : ''}
          </span>
        </div>

        {onRunEvaluation && (
          <button 
            className={`run-eval-primary-btn ${isEvaluating ? 'evaluating' : ''}`}
            onClick={onRunEvaluation}
            disabled={isEvaluating}
          >
            {isEvaluating ? (
              <>
                <span className="eval-spinner"></span>
                <span>Evaluating T-GNN Graph...</span>
              </>
            ) : (
              <>
                <svg className="lightning-icon" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M13 2L3 14h7v8l10-12h-7z" />
                </svg>
                <span>Run {selectedDrugs.length}+ Drug DDI Evaluation</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Medication Dropdown Selection Toolbar */}
      <div className="drug-dropdown-toolbar">
        <div className="dropdown-tool-group">
          <label className="dropdown-label font-bold">
            <span className="lbl-icon">💊</span> Select Drug A:
          </label>
          <select 
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

        <span className="dropdown-vs-badge">VS</span>

        <div className="dropdown-tool-group">
          <label className="dropdown-label font-bold">
            <span className="lbl-icon">💊</span> Select Drug B:
          </label>
          <select 
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
            <label className="dropdown-label font-bold">
              <span className="lbl-icon">⚡</span> Quick DDI Presets:
            </label>
            <select 
              className="med-select-dropdown preset-dropdown"
              defaultValue=""
              onChange={(e) => e.target.value && onSelectPresetPair(e.target.value)}
            >
              <option value="" disabled>-- Choose Common Clinical Pair --</option>
              {PRESET_DRUG_PAIRS.map((preset, idx) => (
                <option key={idx} value={preset.pair.join('+')}>
                  {preset.label}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Selected Medications Grid/List */}
      <div className="selected-meds-container">
        <div className="sub-section-label">
          SELECTED PATIENT MEDICATIONS ({selectedDrugs.length} ACTIVE IN REGIMEN)
        </div>

        <div className={`drug-items-grid ${selectedDrugs.length > 3 ? 'compact-grid' : ''}`}>
          {selectedDrugs.map((drug, index) => {
            const colorScheme = ACCENT_COLORS[index % ACCENT_COLORS.length];
            return (
              <div 
                key={drug.id || index} 
                className={`drug-appointment-item ${colorScheme}-highlight`}
              >
                <div className={`highlight-accent-line ${colorScheme}`}></div>
                <div className="item-content-group">
                  <div className="drug-header-title">
                    <strong className="drug-name-text">{drug.name}</strong>
                    <span className={`code-chip ${colorScheme}`}>{drug.id}</span>
                  </div>
                  <span className="item-subtext">
                    Drug #{index + 1} &bull; {drug.dosage || '40 mg'} {drug.frequency || 'QD'}
                  </span>
                </div>
                <div className="right-checks">
                  <span className="drug-num-badge">{index + 1}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 3+ Drug Pairwise Interaction Matrix */}
      {pairs.length > 0 && (
        <div className="pairwise-matrix-section">
          <div className="matrix-header-row">
            <h4 className="matrix-title">
              Pairwise Drug-Drug Interaction Breakdown ({pairs.length} Pairs Analyzed)
            </h4>
            <div className="matrix-summary-pills">
              {evaluationData?.majorCount > 0 && (
                <span className="matrix-pill major">{evaluationData.majorCount} Major Risk</span>
              )}
              {evaluationData?.modCount > 0 && (
                <span className="matrix-pill moderate">{evaluationData.modCount} Moderate Risk</span>
              )}
              {evaluationData?.lowCount > 0 && (
                <span className="matrix-pill low">{evaluationData.lowCount} Low Risk</span>
              )}
            </div>
          </div>

          <div className="pairwise-cards-list">
            {pairs.map((pair, idx) => (
              <div key={idx} className={`pair-breakdown-card severity-${pair.severity.toLowerCase()}`}>
                <div className="pair-card-top">
                  <div className="pair-drugs-names">
                    <span className="pair-drug-badge">{pair.drug1.name}</span>
                    <span className="pair-arrow">&harr;</span>
                    <span className="pair-drug-badge">{pair.drug2.name}</span>
                  </div>
                  <div className="pair-metrics">
                    <span className={`pair-sev-tag ${pair.severity.toLowerCase()}`}>
                      {pair.severity}
                    </span>
                    <span className="pair-prob-val">
                      {Math.round(pair.probability * 100)}% Risk
                    </span>
                  </div>
                </div>

                {pair.mechanism && (
                  <div className="pair-card-mechanism">
                    <span className="mech-label">Mechanism:</span> {pair.mechanism}
                  </div>
                )}
                {pair.summary && (
                  <p className="pair-card-summary">{pair.summary}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
