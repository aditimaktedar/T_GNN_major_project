import React from 'react';
import './FlowArrow.css';

export function FlowArrow({ label }) {
  return (
    <div className="ehr-flow-arrow-container">
      <div className="ehr-flow-line"></div>
      <div className="ehr-flow-arrow-badge">
        <svg className="down-arrow-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
        {label && <span className="ehr-flow-label">{label}</span>}
      </div>
    </div>
  );
}
