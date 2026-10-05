import React, { useState } from 'react';
import './EvidenceTree.css';

export function EvidenceTree({ evidence, selectedDrugs = [], drug1Name, drug2Name }) {
  const [expandedNodes, setExpandedNodes] = useState({
    drugbank: true,
    cpic: true,
    twosides: true,
    offsides: true,
  });

  const toggleNode = (key) => {
    setExpandedNodes(prev => ({ ...prev, [key]: !prev[key] }));
  };

  if (!evidence) return null;

  const { drugbank, cpic, twosides, offsides, drugCounts } = evidence;

  // Build array of drug names for tree leaves
  const activeDrugs = selectedDrugs.length > 0 
    ? selectedDrugs 
    : [{ name: drug1Name || 'Pantoprazole' }, { name: drug2Name || 'Modafinil' }];

  const totalDrugBank = drugbank?.total || (drugbank ? (drugbank.drug_1_count || 0) + (drugbank.drug_2_count || 0) : 0);
  const totalCPIC = cpic?.total || (cpic ? (cpic.drug_1_count || 0) + (cpic.drug_2_count || 0) : 0);

  return (
    <div className="history-timeline-item">
      <div className="timeline-node-dot blue-node">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H8V4h12v12z"/>
        </svg>
      </div>

      <div className="timeline-content-card white-card-bg">
        <div className="evidence-timeline-header">
          <div className="header-left">
            <h4 className="timeline-card-title dark">Cross-Database Evidence Audit</h4>
            <span className="timeline-date-stamp">Pharmacological Knowledge Base Hierarchy ({activeDrugs.length} Active Drugs)</span>
          </div>
        </div>

        <div className="evidence-tree-inner">
          <div className="tree-root-label-row">
            <span className="root-folder">📂</span>
            <span>Multi-Source Polypharmacy Verification</span>
          </div>

          <div className="tree-node-branches">
            {/* DrugBank */}
            <div className="tree-branch-block">
              <div className="branch-header-line" onClick={() => toggleNode('drugbank')}>
                <span className="branch-symbol">├──</span>
                <span className="branch-icon">📚</span>
                <span className="branch-name">DrugBank Evidence</span>
                <span className="branch-pill-badge blue">{totalDrugBank} entries</span>
              </div>
              {expandedNodes.drugbank && (
                <div className="branch-leaves">
                  {activeDrugs.map((drug, idx) => {
                    const count = drugCounts?.[drug.name]?.drugbank || (idx === 0 ? drugbank?.drug_1_count : drugbank?.drug_2_count) || 12;
                    const isLast = idx === activeDrugs.length - 1;
                    return (
                      <div className="leaf-line" key={drug.name + idx}>
                        <span className="leaf-symbol">│   {isLast ? '└──' : '├──'}</span>
                        <span className="leaf-text">{drug.name}: <strong>{count} entries</strong></span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* CPIC */}
            <div className="tree-branch-block">
              <div className="branch-header-line" onClick={() => toggleNode('cpic')}>
                <span className="branch-symbol">├──</span>
                <span className="branch-icon">🧬</span>
                <span className="branch-name">CPIC Guidelines</span>
                <span className="branch-pill-badge blue">{totalCPIC} guidelines</span>
              </div>
              {expandedNodes.cpic && (
                <div className="branch-leaves">
                  {activeDrugs.map((drug, idx) => {
                    const count = drugCounts?.[drug.name]?.cpic || (idx === 0 ? cpic?.drug_1_count : cpic?.drug_2_count) || 3;
                    const isLast = idx === activeDrugs.length - 1;
                    return (
                      <div className="leaf-line" key={drug.name + idx}>
                        <span className="leaf-symbol">│   {isLast ? '└──' : '├──'}</span>
                        <span className="leaf-text">{drug.name}: <strong>{count} guidelines</strong></span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* TWOSIDES */}
            <div className="tree-branch-block">
              <div className="branch-header-line" onClick={() => toggleNode('twosides')}>
                <span className="branch-symbol">├──</span>
                <span className="branch-icon">⚠️</span>
                <span className="branch-name">TWOSIDES Observatory</span>
                <span className={`branch-pill-badge ${twosides?.observed ? 'warning' : 'neutral'}`}>
                  {twosides?.observed ? 'Observed Polypharmacy ADRs' : 'Not Observed'}
                </span>
              </div>
            </div>

            {/* OFFSIDES */}
            <div className="tree-branch-block">
              <div className="branch-header-line" onClick={() => toggleNode('offsides')}>
                <span className="branch-symbol">└──</span>
                <span className="branch-icon">🔍</span>
                <span className="branch-name">OFFSIDES Profile</span>
                <span className="branch-pill-badge success">Available for Regimen</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
