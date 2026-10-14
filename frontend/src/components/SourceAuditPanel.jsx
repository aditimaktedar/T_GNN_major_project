import React, { useState } from 'react';
import './SourceAuditPanel.css';

export function SourceAuditPanel({ evaluationData, engineHealth, apiStatus }) {
  const [pairsOpen, setPairsOpen] = useState(false);

  const meta = evaluationData?.evaluation_meta;
  const pairs = evaluationData?.pairs || [];
  const evidence = evaluationData?.evidence;
  const prediction = evaluationData?.prediction;

  const engine = engineHealth?.engine || {};

  return (
    <div className="main-panel-step" id="sec-audit">
      <div className="screenshot-card audit-page-card">
        <header className="section-page-header">
          <h2 className="section-page-title">Source audit</h2>
          <p className="section-helper">
            Provenance for the latest evaluation: model identity, retrieval sources, and per-pair
            identifiers returned by the API. Timestamps are not provided by the current backend.
          </p>
        </header>

        {!evaluationData ? (
          <div className="portal-empty-state inline-empty">
            <p className="portal-empty-title">No evaluation to audit</p>
            <p className="portal-empty-copy">
              Run an interaction evaluation first. Audit fields are populated from that response.
            </p>
          </div>
        ) : (
          <>
            <section className="audit-section">
              <h3 className="audit-section-title">Engine &amp; service</h3>
              <dl className="audit-dl">
                <div className="audit-dl-row">
                  <dt>API connection</dt>
                  <dd>
                    {apiStatus === 'connected'
                      ? 'Connected'
                      : apiStatus === 'checking'
                        ? 'Checking…'
                        : 'Offline'}
                  </dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Model (evaluation)</dt>
                  <dd>{meta?.engine || '—'}</dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Model (health endpoint)</dt>
                  <dd>{engine.model || 'Not available until health check succeeds'}</dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Engine status</dt>
                  <dd>{engine.status || '—'}</dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Available CV folds</dt>
                  <dd>
                    {meta?.available_folds?.length
                      ? meta.available_folds.join(', ')
                      : engine.available_folds?.length
                        ? engine.available_folds.join(', ')
                        : '—'}
                  </dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Indexed drugs (RAG index)</dt>
                  <dd>
                    {engine.indexed_drugs_count != null
                      ? String(engine.indexed_drugs_count)
                      : '—'}
                  </dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Evidence chunks indexed</dt>
                  <dd>
                    {engine.indexed_evidence_chunks != null
                      ? String(engine.indexed_evidence_chunks)
                      : '—'}
                  </dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Evaluation timestamp</dt>
                  <dd className="audit-unavailable">Not recorded by API</dd>
                </div>
              </dl>
            </section>

            <section className="audit-section">
              <h3 className="audit-section-title">Regimen prediction record</h3>
              <dl className="audit-dl">
                <div className="audit-dl-row">
                  <dt>Aggregate source label</dt>
                  <dd>{prediction?.source || '—'}</dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Alert code</dt>
                  <dd className="audit-mono">{prediction?.alertCode || '—'}</dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Pairs evaluated</dt>
                  <dd>{meta?.total_pairs_evaluated ?? evaluationData.pairCount ?? '—'}</dd>
                </div>
                <div className="audit-dl-row">
                  <dt>Critical alerts (major)</dt>
                  <dd>{meta?.critical_alerts_count ?? evaluationData.majorCount ?? '—'}</dd>
                </div>
              </dl>
            </section>

            {evidence && (
              <section className="audit-section">
                <h3 className="audit-section-title">Retrieval counts (aggregated)</h3>
                <dl className="audit-dl">
                  <div className="audit-dl-row">
                    <dt>DrugBank entries (total)</dt>
                    <dd>{evidence.drugbank?.total ?? '—'}</dd>
                  </div>
                  <div className="audit-dl-row">
                    <dt>CPIC entries (total)</dt>
                    <dd>{evidence.cpic?.total ?? '—'}</dd>
                  </div>
                  <div className="audit-dl-row">
                    <dt>Twosides signal observed</dt>
                    <dd>{evidence.twosides?.observed ? 'Yes' : 'No'}</dd>
                  </div>
                  <div className="audit-dl-row">
                    <dt>OffSides corpus</dt>
                    <dd>
                      {evidence.offsides?.available === false
                        ? 'Unavailable'
                        : evidence.offsides?.available
                          ? 'Available'
                          : '—'}
                    </dd>
                  </div>
                </dl>
              </section>
            )}

            <section className="audit-section">
              <button
                type="button"
                className="audit-pairs-toggle"
                onClick={() => setPairsOpen((o) => !o)}
                aria-expanded={pairsOpen}
              >
                Per-pair source identifiers ({pairs.length})
              </button>
              {pairsOpen && pairs.length > 0 && (
                <div className="audit-table-wrap">
                  <table className="audit-table">
                    <thead>
                      <tr>
                        <th>Pair</th>
                        <th>Interaction</th>
                        <th>Source</th>
                        <th>Alert code</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pairs.map((pair, idx) => (
                        <tr key={idx}>
                          <td>
                            {pair.drug1?.name} ↔ {pair.drug2?.name}
                          </td>
                          <td>
                            {pair.interaction} · {Math.round((pair.probability ?? 0) * 100)}%
                          </td>
                          <td className="audit-mono">{pair.source || '—'}</td>
                          <td className="audit-mono">{pair.alertCode || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {pairsOpen && pairs.length === 0 && (
                <p className="audit-unavailable">No pairwise records in response.</p>
              )}
            </section>
          </>
        )}
      </div>
    </div>
  );
}
