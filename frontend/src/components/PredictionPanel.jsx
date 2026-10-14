import React, { useState } from 'react';
import './PredictionPanel.css';

function confidenceLabel(probability) {
  const p = probability ?? 0;
  if (p >= 0.75) return 'Higher model score';
  if (p >= 0.5) return 'Moderate model score';
  return 'Lower model score';
}

export function PredictionPanel({ prediction, evaluationData, isEvaluating = false }) {
  const [detailsOpen, setDetailsOpen] = useState(false);

  if (!prediction) return null;

  const { interaction, probability, severity, source } = prediction;
  const percentage = Math.round((probability ?? 0) * 100);

  const selectedDrugs = evaluationData?.selectedDrugs || [];
  const pairCount = evaluationData?.pairCount || 1;
  const highestPair = evaluationData?.highestPair;

  const interactionYes = String(interaction).toUpperCase() === 'YES';

  return (
    <div className={`screenshot-card prediction-card ${isEvaluating ? 'is-refreshing' : ''}`}>
      {isEvaluating && (
        <div className="prediction-refresh-note" role="status">
          Updating results…
        </div>
      )}

      <div className="card-top-title-row">
        <div className="title-with-badge">
          <h3 className="card-main-title">
            {selectedDrugs.length >= 3 ? 'Regimen interaction summary' : 'Interaction summary'}
          </h3>
          <p className="section-helper">
            Model prediction only — confirm with clinical references and local protocols.
          </p>
        </div>
      </div>

      <div className="prediction-boxes-row">
        <div className="prediction-box interaction-result">
          <span className="box-mini-label">Predicted interaction</span>
          <div className={`status-pill-badge ${interactionYes ? 'yes' : 'no'}`}>
            <span className="pill-dot" aria-hidden="true" />
            <span className="pill-text">
              {interactionYes
                ? selectedDrugs.length >= 3
                  ? 'Interaction signal in regimen'
                  : 'Interaction signal detected'
                : 'No interaction signal'}
            </span>
          </div>
          {selectedDrugs.length >= 3 && (
            <span className="poly-pairs-subtext">{pairCount} drug pairs analyzed</span>
          )}
        </div>

        <div className="prediction-box probability-result">
          <span className="box-mini-label">Highest pairwise score</span>
          <div className="prob-text-group">
            <span className="prob-big">{percentage}%</span>
            <span className="prob-confidence">{confidenceLabel(probability)}</span>
          </div>
          <div className="prob-progress-track" role="presentation">
            <div className="prob-progress-bar" style={{ width: `${percentage}%` }} />
          </div>
          <span className="poly-pairs-subtext">Not a probability of clinical harm</span>
        </div>

        <div className="prediction-box severity-result">
          <span className="box-mini-label">Severity label</span>
          <div className={`severity-flag-badge ${severity.toLowerCase()}`}>
            <span className="severity-pulse-dot" aria-hidden="true" />
            <span>{severity}</span>
          </div>
          {highestPair && (
            <span className="highest-pair-subtext">
              Highest-scoring pair: {highestPair.drug1.name} · {highestPair.drug2.name}
            </span>
          )}
        </div>
      </div>

      {source && (
        <div className="prediction-details-block">
          <button
            type="button"
            className="details-toggle-btn"
            onClick={() => setDetailsOpen((o) => !o)}
            aria-expanded={detailsOpen}
          >
            {detailsOpen ? 'Hide technical source' : 'Show technical source'}
          </button>
          {detailsOpen && <p className="prediction-source-line">{source}</p>}
        </div>
      )}
    </div>
  );
}
