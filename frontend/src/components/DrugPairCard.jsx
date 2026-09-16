import React from 'react';
import './DrugPairCard.css';

export function DrugPairCard({ drug1, drug2, onRunEvaluation, isEvaluating }) {
  return (
    <div className="screenshot-card appointments-card">
      <div className="card-top-title-row">
        <h3 className="card-main-title">Evaluated Drug Pair</h3>
        {onRunEvaluation && (
          <button 
            className={`run-eval-primary-btn ${isEvaluating ? 'evaluating' : ''}`}
            onClick={onRunEvaluation}
            disabled={isEvaluating}
          >
            {isEvaluating ? (
              <>
                <span className="eval-spinner"></span>
                <span>Evaluating T-GNN...</span>
              </>
            ) : (
              <>
                <svg className="lightning-icon" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M13 2L3 14h7v8l10-12h-7z" />
                </svg>
                <span>Run DDI Evaluation</span>
              </>
            )}
          </button>
        )}
      </div>

      <div className="sub-section-block">
        <span className="sub-section-label">SELECTED DRUG 1</span>
        <div className="drug-appointment-item blue-highlight">
          <div className="highlight-accent-line blue"></div>
          <div className="item-content-group">
            <div className="drug-header-title">
              <strong className="drug-name-text">{drug1.name}</strong>
              <span className="code-chip blue">{drug1.drugbank_id}</span>
            </div>
            <span className="item-subtext">Primary Target Agent • Administered Oral QD</span>
          </div>
          <div className="right-checks">
            <svg className="double-check-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7M11 13l2 2 4-4" />
            </svg>
            <div className="mini-user-avatar">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
              </svg>
            </div>
          </div>
        </div>
      </div>

      <div className="sub-section-block">
        <span className="sub-section-label">SELECTED DRUG 2</span>
        <div className="drug-appointment-item pink-highlight">
          <div className="highlight-accent-line pink"></div>
          <div className="item-content-group">
            <div className="drug-header-title">
              <strong className="drug-name-text">{drug2.name}</strong>
              <span className="code-chip pink">{drug2.drugbank_id}</span>
            </div>
            <span className="item-subtext">Secondary Co-administered Agent • Administered Oral QAM</span>
          </div>
          <div className="right-checks">
            <svg className="double-check-icon pink" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7M11 13l2 2 4-4" />
            </svg>
            <div className="mini-user-avatar pink">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
