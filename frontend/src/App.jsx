import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { initialPatient, samplePredictionsDatabase } from './data/mockPayload';
import { PatientCard } from './components/PatientCard';
import { DrugPairCard } from './components/DrugPairCard';
import { PredictionPanel } from './components/PredictionPanel';
import { RAGExplanation } from './components/RAGExplanation';
import { EvidenceTree } from './components/EvidenceTree';

export default function App() {
  const [patient, setPatient] = useState(initialPatient);
  const [activeNavTab, setActiveNavTab] = useState('overview');
  const [selectedPair, setSelectedPair] = useState([
    initialPatient.medications[0],
    initialPatient.medications[1],
  ]);

  // Evaluated data state + loading state
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking'); // 'connected' | 'offline' | 'checking'
  const [activeEvaluatedData, setActiveEvaluatedData] = useState(
    samplePredictionsDatabase["Pantoprazole+Modafinil"]
  );

  const [newMedName, setNewMedName] = useState('');
  const [newMedId, setNewMedId] = useState('');
  const [showAddForm, setShowAddForm] = useState(false);

  // Section refs for smooth scrolling navigation
  const overviewRef = useRef(null);
  const medsRef = useRef(null);
  const pairRef = useRef(null);
  const predictionRef = useRef(null);
  const ragRef = useRef(null);
  const evidenceRef = useRef(null);

  // Check backend health on mount
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.ok ? res.json() : Promise.reject())
      .then(() => setApiStatus('connected'))
      .catch(() => setApiStatus('offline'));
  }, []);

  const handleSelectNavTab = (tabId) => {
    setActiveNavTab(tabId);

    const refMap = {
      overview: overviewRef,
      medications: medsRef,
      ddi: predictionRef,
      rag: ragRef,
      evidence: evidenceRef,
    };

    const targetRef = refMap[tabId];
    if (targetRef && targetRef.current) {
      targetRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // Trigger evaluation function with Live API query and fallback
  const handleRunEvaluation = async () => {
    if (selectedPair.length < 2) {
      alert("Please select exactly 2 medications from the MAR table to evaluate.");
      return;
    }

    setIsEvaluating(true);
    const drug1 = selectedPair[0];
    const drug2 = selectedPair[1];

    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          drug_a: drug1.id,
          drug_b: drug2.id,
          drug_a_name: drug1.name,
          drug_b_name: drug2.name,
          patient_id: patient.mrn,
        }),
      });

      if (response.ok) {
        const liveData = await response.json();
        setActiveEvaluatedData(liveData);
        setApiStatus('connected');
        setIsEvaluating(false);

        if (predictionRef.current) {
          predictionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        return;
      }
    } catch (err) {
      console.warn("Backend API not reachable, falling back to local dataset:", err);
      setApiStatus('offline');
    }

    // Fallback demo database logic
    setTimeout(() => {
      const pairKey = `${drug1.name}+${drug2.name}`;
      const pairKeyReverse = `${drug2.name}+${drug1.name}`;

      const matchedData =
        samplePredictionsDatabase[pairKey] ||
        samplePredictionsDatabase[pairKeyReverse] ||
        samplePredictionsDatabase["default"];

      setActiveEvaluatedData(matchedData);
      setIsEvaluating(false);

      if (predictionRef.current) {
        predictionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 450);
  };

  // Update Patient Profile
  const handleUpdatePatient = (updatedProfile) => {
    setPatient((prev) => ({
      ...prev,
      name: updatedProfile.name,
      age: updatedProfile.age,
      gender: updatedProfile.gender || prev.gender,
    }));
  };

  // Add Medication (up to 7 max)
  const handleAddMedication = (e) => {
    e.preventDefault();
    if (!newMedName.trim()) return;
    if (patient.medications.length >= 7) {
      alert("Maximum patient MAR limit of 7 active medications reached.");
      return;
    }
    const drugId = newMedId.trim() || `DB${Math.floor(10000 + Math.random() * 90000)}`;
    const newMed = {
      id: drugId.toUpperCase(),
      name: newMedName.trim(),
      dosage: "10 mg",
      frequency: "QD (Daily)",
      route: "Oral",
    };
    setPatient((prev) => ({
      ...prev,
      medications: [...prev.medications, newMed],
    }));
    setNewMedName('');
    setNewMedId('');
    setShowAddForm(false);
  };

  // Remove Medication
  const handleRemoveMedication = (medId) => {
    if (patient.medications.length <= 2) {
      alert("A minimum of 2 medications is required for drug interaction evaluation.");
      return;
    }
    const updatedMeds = patient.medications.filter((m) => m.id !== medId);
    setPatient((prev) => ({
      ...prev,
      medications: updatedMeds,
    }));

    if (selectedPair.some((m) => m.id === medId)) {
      const remainingSelected = selectedPair.filter((m) => m.id !== medId);
      const replacement = updatedMeds.find((m) => !remainingSelected.some((r) => r.id === m.id));
      if (replacement) {
        setSelectedPair([...remainingSelected, replacement]);
      } else {
        setSelectedPair(updatedMeds.slice(0, 2));
      }
    }
  };

  const toggleSelectDrug = (med) => {
    const isSelected = selectedPair.some(m => m.id === med.id);
    if (isSelected) {
      if (selectedPair.length > 1) {
        setSelectedPair(selectedPair.filter(m => m.id !== med.id));
      }
    } else {
      if (selectedPair.length >= 2) {
        setSelectedPair([selectedPair[1], med]);
      } else {
        setSelectedPair([...selectedPair, med]);
      }
    }
  };

  const drug1 = selectedPair[0] || { id: "DB00213", name: "Pantoprazole" };
  const drug2 = selectedPair[1] || { id: "DB00745", name: "Modafinil" };

  return (
    <div className="app-layout screenshot-theme">
      {/* Top Navbar */}
      <header className="screenshot-top-nav">
        <div className="nav-brand">
          <span className="brand-dot"></span>
          <span className="brand-text font-bold">Clinical Portal &bull; T-GNN DDI Evaluator</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', padding: '4px 12px', borderRadius: '16px', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)' }}>
          <span style={{
            display: 'inline-block',
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: apiStatus === 'connected' ? '#10b981' : '#f59e0b',
            boxShadow: apiStatus === 'connected' ? '0 0 8px #10b981' : 'none'
          }}></span>
          <span style={{ color: '#e2e8f0', fontWeight: 500 }}>
            {apiStatus === 'connected' ? 'Live T-GNN v2 + RAG API' : 'Demo Dataset Mode'}
          </span>
        </div>
      </header>

      {/* Main 2-Column Dashboard Grid */}
      <main className="dashboard-grid-container" ref={overviewRef}>
        {/* Left Column: Patient Profile Sidebar */}
        <aside className="left-sidebar-col">
          <PatientCard 
            patient={patient}
            onUpdatePatient={handleUpdatePatient}
            activeNavTab={activeNavTab}
            onSelectNavTab={handleSelectNavTab}
          />
        </aside>

        {/* Right Column: Main Content Area */}
        <section className="right-main-col">
          
          {/* 1. TOP MOST SECTION: Active Medications & MAR Regimen Table */}
          <div className="main-panel-step" ref={medsRef} id="sec-medications">
            <div className="screenshot-card active-meds-top-card">
              <div className="mar-card-header">
                <div className="header-left">
                  <div className="section-title-with-badge">
                    <h3 className="card-main-title">Active Regimen & MAR Table</h3>
                    <span className="med-count-pill">{patient.medications.length} / 7 Max</span>
                  </div>
                  <span className="timeline-date-stamp">Select any 2 medications below to evaluate DDI risk</span>
                </div>

                <div className="mar-header-actions">
                  {patient.medications.length < 7 && (
                    <button 
                      className="add-med-btn-screenshot"
                      onClick={() => setShowAddForm(!showAddForm)}
                    >
                      {showAddForm ? 'Close' : '+ Add Medication'}
                    </button>
                  )}
                </div>
              </div>

              {showAddForm && (
                <form className="add-med-inline-form" onSubmit={handleAddMedication}>
                  <input 
                    type="text" 
                    placeholder="Drug Name (e.g., Lisinopril)" 
                    value={newMedName}
                    onChange={(e) => setNewMedName(e.target.value)}
                    required
                  />
                  <input 
                    type="text" 
                    placeholder="DrugBank ID (e.g., DB00722)" 
                    value={newMedId}
                    onChange={(e) => setNewMedId(e.target.value)}
                  />
                  <button type="submit" className="confirm-add-blue-btn">Add Order</button>
                </form>
              )}

              <div className="mar-table-wrapper-screenshot">
                <table className="screenshot-mar-table">
                  <thead>
                    <tr>
                      <th>PAIR SELECT</th>
                      <th>DRUGBANK ID</th>
                      <th>MEDICATION NAME</th>
                      <th>DOSAGE</th>
                      <th>FREQUENCY</th>
                      <th>ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patient.medications.map((med) => {
                      const isSelected = selectedPair.some(m => m.id === med.id);
                      return (
                        <tr 
                          key={med.id} 
                          className={isSelected ? 'selected-pair-row' : ''}
                          onClick={() => toggleSelectDrug(med)}
                        >
                          <td className="center-col">
                            <input 
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => toggleSelectDrug(med)}
                              onClick={(e) => e.stopPropagation()}
                            />
                          </td>
                          <td className="mono-code">{med.id}</td>
                          <td className="drug-name-col">
                            <strong>{med.name}</strong>
                            {isSelected && <span className="active-tag">Selected Pair</span>}
                          </td>
                          <td>{med.dosage || '40 mg'}</td>
                          <td>{med.frequency || 'QD (Daily)'}</td>
                          <td className="right-action">
                            {patient.medications.length > 2 && (
                              <button 
                                className="discontinue-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleRemoveMedication(med.id);
                                }}
                              >
                                Discontinue
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* 2. Evaluated Drug Pair Section */}
          <div className="main-panel-step" ref={pairRef} id="sec-pair">
            <DrugPairCard 
              drug1={{ drugbank_id: drug1.id, name: drug1.name }}
              drug2={{ drugbank_id: drug2.id, name: drug2.name }}
              onRunEvaluation={handleRunEvaluation}
              isEvaluating={isEvaluating}
            />
          </div>

          {/* 3. DDI Risk Prediction Panel */}
          <div className="main-panel-step" ref={predictionRef} id="sec-prediction">
            <PredictionPanel 
              prediction={activeEvaluatedData.prediction} 
              isEvaluating={isEvaluating}
            />
          </div>

          {/* 4. History & Clinical Evidence Container */}
          <div className="screenshot-card history-section-card">
            <div className="history-card-header">
              <h3 className="card-main-title">History & Clinical Evidence</h3>
            </div>

            <div className="history-timeline-container">
              <div className="timeline-vertical-line"></div>

              {/* RAG Explanation Timeline Node */}
              <div ref={ragRef} id="sec-rag">
                <RAGExplanation ragData={activeEvaluatedData.rag_explanation} />
              </div>

              {/* Multi-Database Evidence Tree Timeline Node */}
              <div ref={evidenceRef} id="sec-evidence">
                <EvidenceTree 
                  evidence={activeEvaluatedData.evidence}
                  drug1Name={drug1.name}
                  drug2Name={drug2.name}
                />
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
