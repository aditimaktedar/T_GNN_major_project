import React from 'react';
import './PredictionPanel.css';

export function PredictionPanel({ prediction }) {
  const { interaction, probability, severity, source } = prediction;
  const percentage = Math.round(probability * 100);

  return (
    <div className="screenshot-card prediction-card">
      <div className="card-top-title-row">
        <div className="title-with-badge">
          <h3 className="card-main-title">DDI Interaction Risk Prediction</h3>
          <span className="gnn-source-tag">{source || 'T-GNN Model'}</span>
        </div>
      </div>

      <div className="prediction-boxes-row">
        {/* Interaction Status */}
        <div className="prediction-box interaction-result">
          <span className="box-mini-label">INTERACTION STATUS</span>
          <div className={`status-pill-badge ${interaction.toLowerCase()}`}>
            <span className="pill-dot"></span>
            <span className="pill-text">{interaction === 'YES' ? 'YES — INTERACTION DETECTED' : 'NO INTERACTION'}</span>
          </div>
        </div>

        {/* Probability */}
        <div className="prediction-box probability-result">
          <span className="box-mini-label">MODEL PROBABILITY</span>
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
          <span className="box-mini-label">SEVERITY LEVEL</span>
          <div className={`severity-flag-badge ${severity.toLowerCase()}`}>
            <span className="severity-pulse-dot"></span>
            <span>{severity} SEVERITY</span>
          </div>
        </div>
      </div>
    </div>
  );
}
