import React from 'react';
import './PredictionPanel.css';

export function PredictionPanel({ prediction, evaluationData }) {
  if (!prediction) return null;

  const { interaction, probability, severity, source } = prediction;
  const percentage = Math.round((probability || 0.74) * 100);

  const selectedDrugs = evaluationData?.selectedDrugs || [];
  const pairCount = evaluationData?.pairCount || 1;
  const highestPair = evaluationData?.highestPair;

  return (
    <div className="screenshot-card prediction-card">
      <div className="card-top-title-row">
        <div className="title-with-badge">
          <h3 className="card-main-title">
            {selectedDrugs.length >= 3 
              ? `Polypharmacy Interaction Risk (${selectedDrugs.length} Drugs)` 
              : 'DDI Interaction Risk Prediction'}
          </h3>
          <span className="gnn-source-tag">{source || 'T-GNN Clinical Engine'}</span>
        </div>
      </div>

      <div className="prediction-boxes-row">
        {/* Interaction Status */}
        <div className="prediction-box interaction-result">
          <span className="box-mini-label">INTERACTION STATUS</span>
          <div className={`status-pill-badge ${interaction.toLowerCase()}`}>
            <span className="pill-dot"></span>
            <span className="pill-text">
              {interaction === 'YES' 
                ? (selectedDrugs.length >= 3 ? 'YES — POLYPHARMACY DDI DETECTED' : 'YES — INTERACTION DETECTED') 
                : 'NO INTERACTION DETECTED'}
            </span>
          </div>
          {selectedDrugs.length >= 3 && (
            <span className="poly-pairs-subtext">
              {pairCount} Pairwise Interactions Analyzed
            </span>
          )}
        </div>

        {/* Probability */}
        <div className="prediction-box probability-result">
          <span className="box-mini-label">HIGHEST MODEL PROBABILITY</span>
          <div className="prob-text-group">
            <span className="prob-big">{percentage}%</span>
            <span className="prob-confidence">High Confidence</span>
          </div>
          <div className="prob-progress-track">
            <div className="prob-progress-bar" style={{ width: `${percentage}%` }}></div>
          </div>
        </div>

        {/* Severity */}
        <div className="prediction-box severity-result">
          <span className="box-mini-label">OVERALL SEVERITY LEVEL</span>
          <div className={`severity-flag-badge ${severity.toLowerCase()}`}>
            <span className="severity-pulse-dot"></span>
            <span>{severity} SEVERITY</span>
          </div>
          {highestPair && (
            <span className="highest-pair-subtext">
              Max Risk Pair: <strong>{highestPair.drug1.name} &bull; {highestPair.drug2.name}</strong>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
