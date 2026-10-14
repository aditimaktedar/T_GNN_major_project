import React from 'react';
import './OverviewSection.css';

export function OverviewSection({
  patient,
  selectedDrugs,
  evaluationData,
  apiStatus,
  isLoadingDB,
  evaluationError,
  onGoToSection,
}) {
  const pred = evaluationData?.prediction;
  const hasEval = Boolean(evaluationData && pred);

  let evalStatusLabel = 'No evaluation run yet';
  let evalStatusDetail =
    'Select medications and run an interaction evaluation to see model output.';

  if (isLoadingDB) {
    evalStatusLabel = 'Loading';
    evalStatusDetail = 'Fetching patient regimen and initial evaluation from the server.';
  } else if (evaluationError) {
    evalStatusLabel = 'Evaluation unavailable';
    evalStatusDetail = evaluationError;
  } else if (hasEval) {
    const pct = Math.round((pred.probability ?? 0) * 100);
    evalStatusLabel = `${pred.interaction === 'YES' ? 'Interaction signal' : 'No interaction signal'} · ${pred.severity} severity`;
    evalStatusDetail = `Latest run: ${evaluationData.pairCount ?? 0} pairwise checks, highest score ${pct}%.`;
  }

  return (
    <div className="main-panel-step" id="sec-overview">
      <div className="screenshot-card overview-page-card">
        <header className="section-page-header">
          <h2 className="section-page-title">Overview</h2>
          <p className="section-helper">
            Summary of this patient&apos;s profile, regimen size, and the most recent interaction evaluation.
          </p>
        </header>

        <div className="overview-grid">
          <div className="overview-stat-block">
            <span className="overview-stat-label">Patient</span>
            <span className="overview-stat-value">{patient.name}</span>
            <span className="overview-stat-sub">
              {patient.gender}, {patient.age} years
              {patient.mrn ? ` · ${patient.mrn}` : ''}
            </span>
          </div>

          <div className="overview-stat-block">
            <span className="overview-stat-label">Active medications</span>
            <span className="overview-stat-value">{patient.medications.length}</span>
            <span className="overview-stat-sub">
              {selectedDrugs.length} selected for evaluation
            </span>
          </div>

          <div className="overview-stat-block">
            <span className="overview-stat-label">Model service</span>
            <span className="overview-stat-value">
              {apiStatus === 'connected'
                ? 'Connected'
                : apiStatus === 'checking'
                  ? 'Checking…'
                  : 'Offline'}
            </span>
            <span className="overview-stat-sub">Backend API for predictions and RAG</span>
          </div>

          <div className="overview-stat-block overview-stat-wide">
            <span className="overview-stat-label">Latest evaluation</span>
            <span className="overview-stat-value">{evalStatusLabel}</span>
            <span className="overview-stat-sub">{evalStatusDetail}</span>
          </div>
        </div>

        {hasEval && selectedDrugs.length > 0 && (
          <div className="overview-regimen-preview">
            <span className="overview-stat-label">Current evaluation regimen</span>
            <p className="overview-regimen-names">
              {selectedDrugs.map((d) => d.name).join(' · ')}
            </p>
          </div>
        )}

        <div className="overview-quick-links">
          <span className="overview-stat-label">Go to</span>
          <div className="overview-link-row">
            <button type="button" className="overview-link-btn" onClick={() => onGoToSection('medications')}>
              Medications
            </button>
            <button type="button" className="overview-link-btn" onClick={() => onGoToSection('ddi')}>
              Interaction results
            </button>
            {hasEval && (
              <>
                <button type="button" className="overview-link-btn" onClick={() => onGoToSection('rag')}>
                  Evidence &amp; explanation
                </button>
                <button type="button" className="overview-link-btn" onClick={() => onGoToSection('evidence')}>
                  Source audit
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
