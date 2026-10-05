import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { PatientCard } from './components/PatientCard';
import { DrugPairCard } from './components/DrugPairCard';
import { PredictionPanel } from './components/PredictionPanel';
import { RAGExplanation } from './components/RAGExplanation';
import { EvidenceTree } from './components/EvidenceTree';
import { 
  fetchPatientProfile, 
  updatePatientProfile, 
  addMedicationOrder, 
  discontinueMedicationOrder, 
  evaluateRegimenDDI 
} from './api/client';

const MAX_MEDICATIONS = 15;

const fallbackPatient = {
  mrn: "MRN-9842-7019",
  name: "Eleanor Vance",
  firstName: "Eleanor",
  lastName: "Vance",
  initials: "EV",
  age: 72,
  dob: "1954-07-22",
  gender: "Female",
  language: "English",
  phone: "(406) 555-0120",
  emergencyContact: "(480) 555-0103",
  email: "eleanor.vance@example.com",
  address: "2972 Westheimer Rd. Santa Ana, Illinois 85486",
  weight: "68 kg",
  room: "Bed 304-B (Step-down Unit)",
  physician: "Dr. Jordan Hughes",
  medications: [
    { id: "DB00213", name: "Pantoprazole", dosage: "40 mg", frequency: "QD (Daily)", route: "Oral" },
    { id: "DB00745", name: "Modafinil", dosage: "200 mg", frequency: "QAM (Morning)", route: "Oral" },
    { id: "DB00722", name: "Lisinopril", dosage: "10 mg", frequency: "QD (Daily)", route: "Oral" },
    { id: "DB00331", name: "Metformin", dosage: "500 mg", frequency: "BID (Twice Daily)", route: "Oral" },
    { id: "DB01076", name: "Atorvastatin", dosage: "20 mg", frequency: "QHS (Bedtime)", route: "Oral" },
    { id: "DB00537", name: "Ciprofloxacin", dosage: "250 mg", frequency: "BID", route: "Oral" },
    { id: "DB00381", name: "Amlodipine", dosage: "5 mg", frequency: "QD", route: "Oral" },
  ],
};

export default function App() {
  const [patient, setPatient] = useState(fallbackPatient);
  const [activeNavTab, setActiveNavTab] = useState('overview');
  const [activeSegment, setActiveSegment] = useState('medications');
  const [isLoadingDB, setIsLoadingDB] = useState(true);

  // Dark Mode state with localStorage persistence
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('tgnn_theme') === 'dark';
  });

  const toggleDarkMode = () => {
    setDarkMode((prev) => {
      const next = !prev;
      localStorage.setItem('tgnn_theme', next ? 'dark' : 'light');
      return next;
    });
  };

  // 3+ Drug Selection State
  const [selectedDrugs, setSelectedDrugs] = useState([]);

  // Evaluated data state + loading state
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [activeEvaluatedData, setActiveEvaluatedData] = useState(null);

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

  // Load Patient Profile and Active Regimen from Backend Database API
  useEffect(() => {
    async function loadDatabaseData() {
      setIsLoadingDB(true);
      try {
        const dbPatient = await fetchPatientProfile();
        if (dbPatient && dbPatient.medications) {
          setPatient(dbPatient);
          const initialSelected = dbPatient.medications.slice(0, 3);
          setSelectedDrugs(initialSelected);

          // Initial evaluation query to database
          const evalResult = await evaluateRegimenDDI(initialSelected);
          setActiveEvaluatedData(evalResult);
        }
      } catch (err) {
        console.warn("Could not load from API server, initializing default state:", err);
        setSelectedDrugs(fallbackPatient.medications.slice(0, 3));
      } finally {
        setIsLoadingDB(false);
      }
    }
    loadDatabaseData();
  }, []);

  const handleSelectNavTab = (tabId) => {
    setActiveNavTab(tabId);

    if (tabId === 'overview' || tabId === 'medications' || tabId === 'ddi') {
      setActiveSegment('medications');
    } else if (tabId === 'rag' || tabId === 'evidence') {
      setActiveSegment('rag');
    }

    const refMap = {
      overview: medsRef,
      medications: medsRef,
      ddi: predictionRef,
      rag: ragRef,
      evidence: evidenceRef,
    };

    setTimeout(() => {
      const targetRef = refMap[tabId];
      if (targetRef && targetRef.current) {
        targetRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);
  };

  // Trigger polypharmacy DDI evaluation via Database API
  const handleRunEvaluation = async () => {
    if (selectedDrugs.length < 2) {
      alert("Please select at least 2 medications from the MAR table to run DDI interaction evaluation.");
      return;
    }

    setIsEvaluating(true);

    try {
      const evaluationResult = await evaluateRegimenDDI(selectedDrugs);
      setActiveEvaluatedData(evaluationResult);
    } catch (err) {
      console.error("Evaluation API failed:", err);
      alert("Evaluation request failed. Please check backend database connectivity.");
    } finally {
      setIsEvaluating(false);

      // Scroll to prediction results
      if (predictionRef.current) {
        predictionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  // Update Patient Profile in Database
  const handleUpdatePatient = async (updatedProfile) => {
    try {
      const updated = await updatePatientProfile(updatedProfile);
      setPatient((prev) => ({
        ...prev,
        name: updated.name || updatedProfile.name,
        age: updated.age || updatedProfile.age,
        gender: updated.gender || updatedProfile.gender || prev.gender,
      }));
    } catch (err) {
      console.error("Failed to update patient profile in DB:", err);
      setPatient((prev) => ({
        ...prev,
        name: updatedProfile.name,
        age: updatedProfile.age,
        gender: updatedProfile.gender || prev.gender,
      }));
    }
  };

  // Add Medication Order to Database (Supports up to 15 max model capacity)
  const handleAddMedication = async (e) => {
    e.preventDefault();
    if (!newMedName.trim()) return;
    if (patient.medications.length >= MAX_MEDICATIONS) {
      alert(`Maximum T-GNN model capacity of ${MAX_MEDICATIONS} active medications reached.`);
      return;
    }

    const drugId = newMedId.trim() || `DB${Math.floor(10000 + Math.random() * 90000)}`;
    const newMedPayload = {
      drugbank_id: drugId.toUpperCase(),
      name: newMedName.trim(),
      dosage: "10 mg",
      frequency: "QD (Daily)",
      route: "Oral",
    };

    try {
      const addedMed = await addMedicationOrder(newMedPayload);
      const formattedMed = {
        id: addedMed.drugbank_id || addedMed.id,
        db_id: addedMed.id,
        name: addedMed.name,
        dosage: addedMed.dosage,
        frequency: addedMed.frequency,
        route: addedMed.route,
      };

      setPatient((prev) => ({
        ...prev,
        medications: [...prev.medications, formattedMed],
      }));
      // Auto select newly added medication for evaluation
      setSelectedDrugs((prev) => [...prev, formattedMed]);
      setNewMedName('');
      setNewMedId('');
      setShowAddForm(false);
    } catch (err) {
      console.error("Error adding medication to DB:", err);
      alert(`Error adding medication: ${err.message}`);
    }
  };

  // Remove Medication Order from Database
  const handleRemoveMedication = async (medId) => {
    if (patient.medications.length <= 2) {
      alert("A minimum of 2 medications is required in the patient regimen for DDI analysis.");
      return;
    }

    const targetMed = patient.medications.find(m => m.id === medId || m.db_id === medId);
    const numericId = targetMed?.db_id || medId;

    try {
      await discontinueMedicationOrder(numericId);
      const updatedMeds = patient.medications.filter((m) => m.id !== medId && m.db_id !== numericId);
      setPatient((prev) => ({
        ...prev,
        medications: updatedMeds,
      }));

      if (selectedDrugs.some((m) => m.id === medId || m.db_id === numericId)) {
        const updatedSelected = selectedDrugs.filter((m) => m.id !== medId && m.db_id !== numericId);
        if (updatedSelected.length >= 2) {
          setSelectedDrugs(updatedSelected);
        } else {
          setSelectedDrugs(updatedMeds.slice(0, 2));
        }
      }
    } catch (err) {
      console.error("Error discontinuing medication in DB:", err);
      // Fallback local update
      const updatedMeds = patient.medications.filter((m) => m.id !== medId);
      setPatient((prev) => ({ ...prev, medications: updatedMeds }));
    }
  };

  // Toggle selection for 3+ multi-drug evaluation
  const toggleSelectDrug = (med) => {
    const isSelected = selectedDrugs.some((m) => m.id === med.id);
    if (isSelected) {
      if (selectedDrugs.length <= 2) {
        alert("At least 2 medications must remain selected for DDI interaction testing.");
        return;
      }
      setSelectedDrugs(selectedDrugs.filter((m) => m.id !== med.id));
    } else {
      setSelectedDrugs([...selectedDrugs, med]);
    }
  };

  // Select All / Deselect All Helper
  const handleSelectAllToggle = () => {
    if (selectedDrugs.length === patient.medications.length) {
      setSelectedDrugs(patient.medications.slice(0, 2));
    } else {
      setSelectedDrugs([...patient.medications]);
    }
  };

  const showMedicationsSegment = activeSegment === 'medications' || activeSegment === 'all';
  const showRAGSegment = activeSegment === 'rag' || activeSegment === 'all';

  return (
    <div className={`app-layout screenshot-theme ${darkMode ? 'dark-theme' : ''}`}>
      {/* Top Navbar */}
      <header className="screenshot-top-nav">
        <div className="nav-brand">
          <span className="brand-dot"></span>
          <span className="brand-text">Clinical Portal &bull; T-GNN Polypharmacy DDI Evaluator</span>
          <span className="db-status-badge">
            <span className="live-db-dot"></span>
          </span>
        </div>

        <div className="nav-actions">
          {/* Small Dark Mode Toggle Button */}
          <button 
            className="theme-toggle-btn"
            onClick={toggleDarkMode}
            title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
            aria-label="Toggle theme mode"
          >
            {darkMode ? (
              <>
                <svg className="theme-icon sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="5" />
                  <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" strokeLinecap="round" />
                </svg>
                <span>Light</span>
              </>
            ) : (
              <>
                <svg className="theme-icon moon-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
                <span>Dark</span>
              </>
            )}
          </button>
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
          
          {/* SEGMENTED CONTROL HEADER */}
          <div className="segmented-control-bar">
            <div className="segmented-control">
              <button 
                className={`segment-tab ${activeSegment === 'medications' ? 'active' : ''}`}
                onClick={() => setActiveSegment('medications')}
              >
                <span className="segment-icon">💊</span>
                <span className="segment-title">Medications & DDI Evaluation</span>
                <span className="segment-badge">{patient.medications.length} Meds</span>
              </button>

              <button 
                className={`segment-tab ${activeSegment === 'rag' ? 'active' : ''}`}
                onClick={() => setActiveSegment('rag')}
              >
                <span className="segment-icon">🔬</span>
                <span className="segment-title">RAG Implementation & Evidence</span>
                
              </button>

              <button 
                className={`segment-tab ${activeSegment === 'all' ? 'active' : ''}`}
                onClick={() => setActiveSegment('all')}
              >
                <span className="segment-icon">📊</span>
                <span className="segment-title">All Sections</span>
              </button>
            </div>
          </div>

          {/* SEGMENT 1: MEDICATIONS & DDI EVALUATION */}
          {showMedicationsSegment && (
            <>
              {/* 1. TOP MOST SECTION: Active Medications & MAR Regimen Table */}
              <div className="main-panel-step" ref={medsRef} id="sec-medications">
                <div className="screenshot-card active-meds-top-card">
                  <div className="mar-card-header">
                    <div className="header-left">
                      <div className="section-title-with-badge">
                        <h3 className="card-main-title">Active Regimen & MAR Table</h3>
                        <span className="med-count-pill">
                          {patient.medications.length} / {MAX_MEDICATIONS} Max (Database Capacity)
                        </span>
                      </div>
                      <span className="timeline-date-stamp">
                        Fetched from SQLite Database • Select 2, 3+ medications below to evaluate polypharmacy DDI risk
                      </span>
                    </div>

                    <div className="mar-header-actions">
                      <button 
                        className="select-all-btn"
                        onClick={handleSelectAllToggle}
                      >
                        {selectedDrugs.length === patient.medications.length ? 'Deselect Extra' : `Select All (${patient.medications.length})`}
                      </button>

                      {patient.medications.length < MAX_MEDICATIONS && (
                        <button 
                          className="add-med-btn-screenshot"
                          onClick={() => setShowAddForm(!showAddForm)}
                        >
                          {showAddForm ? 'Close Form' : '+ Add Medication Order'}
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
                      <button type="submit" className="confirm-add-blue-btn">Save to DB</button>
                    </form>
                  )}

                  <div className="mar-table-wrapper-screenshot">
                    <table className="screenshot-mar-table">
                      <thead>
                        <tr>
                          <th>TEST SELECT</th>
                          <th>DRUGBANK ID</th>
                          <th>MEDICATION NAME</th>
                          <th>DOSAGE</th>
                          <th>FREQUENCY</th>
                          <th>ACTION</th>
                        </tr>
                      </thead>
                      <tbody>
                        {patient.medications.map((med) => {
                          const isSelected = selectedDrugs.some((m) => m.id === med.id || m.name === med.name);
                          return (
                            <tr 
                              key={med.id || med.drugbank_id} 
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
                              <td className="mono-code">{med.id || med.drugbank_id}</td>
                              <td className="drug-name-col">
                                <strong>{med.name}</strong>
                                {isSelected && <span className="active-tag">Testing</span>}
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

              {/* 2. Evaluated Multi-Drug Regimen Section */}
              <div className="main-panel-step" ref={pairRef} id="sec-pair">
                <DrugPairCard 
                  selectedDrugs={selectedDrugs}
                  evaluationData={activeEvaluatedData}
                  onRunEvaluation={handleRunEvaluation}
                  isEvaluating={isEvaluating}
                />
              </div>

              {/* 3. DDI Risk Prediction Panel */}
              {activeEvaluatedData && (
                <div className="main-panel-step" ref={predictionRef} id="sec-prediction">
                  <PredictionPanel 
                    prediction={activeEvaluatedData.prediction}
                    evaluationData={activeEvaluatedData}
                    isEvaluating={isEvaluating}
                  />
                </div>
              )}
            </>
          )}

          {/* SEGMENT 2: RAG IMPLEMENTATION & CLINICAL EVIDENCE */}
          {showRAGSegment && activeEvaluatedData && (
            <div className="main-panel-step" ref={ragRef} id="sec-rag">
              {/* Quick Regimen Context Banner when viewing RAG segment */}
              {activeSegment === 'rag' && (
                <div className="screenshot-card rag-context-banner">
                  <div className="banner-left">
                    <span className="banner-icon">🔬</span>
                    <div className="banner-text">
                      <strong>Active RAG Regimen Context:</strong> {selectedDrugs.map(d => d.name).join(' + ')}
                    </div>
                  </div>
                  <span className={`banner-sev-pill ${activeEvaluatedData?.prediction?.severity?.toLowerCase()}`}>
                    {activeEvaluatedData?.prediction?.severity} SEVERITY ({Math.round((activeEvaluatedData?.prediction?.probability || 0) * 100)}% Risk)
                  </span>
                </div>
              )}

              {/* 4. History & Clinical Evidence Container */}
              <div className="screenshot-card history-section-card">
                <div className="history-card-header">
                  <h3 className="card-main-title">History & Clinical Evidence</h3>
                </div>

                <div className="history-timeline-container">
                  <div className="timeline-vertical-line"></div>

                  {/* RAG Explanation Timeline Node */}
                  <div>
                    <RAGExplanation ragData={activeEvaluatedData.rag_explanation} />
                  </div>

                  {/* Multi-Database Evidence Tree Timeline Node */}
                  <div ref={evidenceRef} id="sec-evidence">
                    <EvidenceTree 
                      evidence={activeEvaluatedData.evidence}
                      selectedDrugs={selectedDrugs}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

        </section>
      </main>
    </div>
  );
}
