import React, { useState } from 'react';
import './RAGExplanation.css';

export function RAGExplanation({ ragData }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!ragData) return null;

  const { summary, shared_pk, shared_targets, shared_mechanisms, shared_pd_effects } = ragData;

  return (
    <div className="history-timeline-item">
      <div className="timeline-node-dot neutral-node">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
        </svg>
      </div>

      <div className="timeline-content-card neutral-card-bg">
        <button type="button" className="timeline-card-header" onClick={() => setIsOpen(!isOpen)} aria-expanded={isOpen}>
          <div className="header-left">
            <h4 className="timeline-card-title">Retrieved mechanism context (RAG)</h4>
            <span className="timeline-date-stamp">Supporting reference text, not a clinical verdict</span>
          </div>
          <svg className={`chevron-toggle ${isOpen ? 'open' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {isOpen && (
          <div className="timeline-card-body">
            {summary && (
              <p className="rag-summary-text">{summary}</p>
            )}

            {/* Table layout matching the screenshot's CODE - DESCRIPTION table */}
            <div className="rag-table-wrapper">
              <table className="rag-screenshot-table">
                <thead>
                  <tr>
                    <th>PROPERTY TYPE</th>
                    <th>IDENTIFIED PATHWAY / FEATURE</th>
                    <th>CLINICAL EFFECT</th>
                  </tr>
                </thead>
                <tbody>
                  {shared_pk && shared_pk.map((item, idx) => (
                    <tr key={`pk-${idx}`}>
                      <td className="code-cell-text">PK - {item.toUpperCase()}</td>
                      <td>Hepatic CYP Metabolism Overlap</td>
                      <td className="tag-val-text green">Altered Drug Bioavailability</td>
                    </tr>
                  ))}
                  {shared_mechanisms && shared_mechanisms.map((item, idx) => (
                    <tr key={`mech-${idx}`}>
                      <td className="code-cell-text">MECHANISM</td>
                      <td>{item}</td>
                      <td className="tag-val-text purple">Enzyme Inhibition Risk</td>
                    </tr>
                  ))}
                  {shared_targets && shared_targets.map((item, idx) => (
                    <tr key={`tgt-${idx}`}>
                      <td className="code-cell-text">TARGET</td>
                      <td>{item}</td>
                      <td className="tag-val-text blue">Receptor Competition</td>
                    </tr>
                  ))}
                  {shared_pd_effects && shared_pd_effects.map((item, idx) => (
                    <tr key={`pd-${idx}`}>
                      <td className="code-cell-text">PD EFFECT</td>
                      <td>{item}</td>
                      <td className="tag-val-text red">Adverse Event Potential</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
