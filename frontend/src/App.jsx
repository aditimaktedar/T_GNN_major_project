import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { PatientCard } from './components/PatientCard';
import { DrugPairCard } from './components/DrugPairCard';
import { PredictionPanel } from './components/PredictionPanel';
import { RAGExplanation } from './components/RAGExplanation';
import { EvidenceTree } from './components/EvidenceTree';
import { COMMON_MEDICATIONS } from './data/commonMedications';
import {
  fetchPatientProfile,
  updatePatientProfile,
  addMedicationOrder,
  discontinueMedicationOrder,
  evaluateRegimenDDI,
  fetchHealthStatus,
} from './api/client';
import { RagIcon } from './components/SegmentIcons';
import { calculateAgeFromDob } from './components/dobDateUtils';
import { generatePdfReport } from './utils/generatePdfReport';
import { OverviewSection } from './components/OverviewSection';
import { SourceAuditPanel } from './components/SourceAuditPanel';

const MAX_MEDICATIONS = 15;
const NAV_SECTIONS = ['overview', 'medications', 'ddi', 'rag', 'evidence'];

function normalizePatientFromApi(raw) {
  if (!raw) return raw;
  const dob = raw.dob;
  let age = raw.age;
  if (dob) {
    const derived = calculateAgeFromDob(dob);
    if (derived != null) {
      age = derived;
    }
  }
  return {
    ...raw,
    mrn: raw.mrn,
    name: raw.name,
    firstName: raw.first_name ?? raw.firstName,
    lastName: raw.last_name ?? raw.lastName,
    initials: raw.initials,
    age,
    dob,
    gender: raw.gender,
    language: raw.language,
    weight: raw.weight,
    room: raw.room,
    physician: raw.physician,
    phone: raw.phone,
    email: raw.email,
    address: raw.address,
    emergencyContact: raw.emergency_contact ?? raw.emergencyContact,
    medications: Array.isArray(raw.medications) ? raw.medications : [],
  };
}

function selectionSignature(drugs) {
  return drugs
    .map((d) => d.id || d.drugbank_id || d.name || '')
    .filter(Boolean)
    .sort()
    .join('|');
}

function DdiEvaluationResults({ evaluationData, isEvaluating, resultsStale, resultsAnchorRef }) {
  if (resultsStale) {
    return (
      <div className="screenshot-card portal-empty-state">
        <p className="portal-empty-title">Selection changed</p>
        <p className="portal-empty-copy">
          Run interaction evaluation again to refresh results for the current medications.
        </p>
      </div>
    );
  }

  if (!evaluationData) {
    return (
      <div className="screenshot-card portal-empty-state">
        <p className="portal-empty-title">No evaluation results yet</p>
        <p className="portal-empty-copy">
          Select at least two medications above, then run interaction evaluation.
        </p>
      </div>
    );
  }

  const pairs = evaluationData.pairs || [];

  return (
    <div className="ddi-results-stack" ref={resultsAnchorRef} id="ddi-evaluation-output">
      <PredictionPanel
        prediction={evaluationData.prediction}
        evaluationData={evaluationData}
        isEvaluating={isEvaluating}
      />

      {pairs.length > 0 && (
        <div className="screenshot-card ddi-pairwise-card">
          <h3 className="card-main-title">Pairwise interaction results</h3>
          <p className="section-helper">
            Individual pair scores from the model for this evaluation ({pairs.length} pair
            {pairs.length === 1 ? '' : 's'}).
          </p>
          <div className="pairwise-cards-list ddi-pairwise-list">
            {pairs.map((pair, idx) => (
              <div
                key={`${pair.drug1?.id}-${pair.drug2?.id}-${idx}`}
                className={`pair-breakdown-card severity-${(pair.severity || 'low').toLowerCase()}`}
              >
                <div className="pair-card-top">
                  <div className="pair-drugs-names">
                    <span className="pair-drug-badge">{pair.drug1?.name}</span>
                    <span className="pair-arrow" aria-hidden="true">
                      ↔
                    </span>
                    <span className="pair-drug-badge">{pair.drug2?.name}</span>
                  </div>
                  <div className="pair-metrics">
                    {pair.severity && (
                      <span className={`pair-sev-tag ${pair.severity.toLowerCase()}`}>{pair.severity}</span>
                    )}
                    {pair.probability != null && (
                      <span className="pair-prob-val">
                        {Math.round(pair.probability * 100)}% model score
                      </span>
                    )}
                    {pair.interaction && (
                      <span className="pair-interaction-tag">{pair.interaction}</span>
                    )}
                  </div>
                </div>
                {pair.mechanism && (
                  <div className="pair-card-mechanism">
                    <span className="mech-label">Mechanism:</span> {pair.mechanism}
                  </div>
                )}
                {pair.summary && <p className="pair-card-summary">{pair.summary}</p>}
              </div>
            ))}
          </div>
        </div>
      )}

      {evaluationData.rag_explanation && (
        <div className="screenshot-card ddi-rag-card">
          <RAGExplanation ragData={evaluationData.rag_explanation} />
        </div>
      )}
    </div>
  );
}

function sectionFromHash() {
  const id = window.location.hash.replace(/^#/, '');
  return NAV_SECTIONS.includes(id) ? id : 'overview';
}
 
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
  const [activeNavTab, setActiveNavTab] = useState(() =>
    typeof window !== 'undefined' ? sectionFromHash() : 'overview'
  );
  const [isLoadingDB, setIsLoadingDB] = useState(true);
  const [engineHealth, setEngineHealth] = useState(null);
 
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
  const [apiStatus, setApiStatus] = useState('checking'); // 'connected' | 'offline' | 'checking'
  const [activeEvaluatedData, setActiveEvaluatedData] = useState(null);
  const [evaluationError, setEvaluationError] = useState(null);
  const [loadNotice, setLoadNotice] = useState(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [reportError, setReportError] = useState(null);
  const evaluationAbortRef = useRef(null);
  const ddiResultsAnchorRef = useRef(null);
  const scrollToResultsPendingRef = useRef(false);
  const mainContentRef = useRef(null);
 
  const [newMedName, setNewMedName] = useState('');
  const [newMedId, setNewMedId] = useState('');
  const [showAddForm, setShowAddForm] = useState(false);
 
  // Check backend health on mount
  useEffect(() => {
    fetchHealthStatus()
      .then((health) => {
        setEngineHealth(health);
        setApiStatus(health?.engine?.status === 'healthy' ? 'connected' : 'connected');
      })
      .catch(() => setApiStatus('offline'));
  }, []);

  useEffect(() => {
    const syncFromHash = () => setActiveNavTab(sectionFromHash());
    window.addEventListener('hashchange', syncFromHash);
    window.addEventListener('popstate', syncFromHash);
    if (!window.location.hash) {
      window.history.replaceState({ section: 'overview' }, '', '#overview');
    }
    return () => {
      window.removeEventListener('hashchange', syncFromHash);
      window.removeEventListener('popstate', syncFromHash);
    };
  }, []);
 
  // Load Patient Profile and Active Regimen from Backend Database API
  useEffect(() => {
    async function loadDatabaseData() {
      setIsLoadingDB(true);
      try {
        const dbPatient = await fetchPatientProfile();
        if (dbPatient && dbPatient.medications) {
          setPatient(normalizePatientFromApi(dbPatient));
          const initialSelected = dbPatient.medications.slice(0, 3);
          setSelectedDrugs(initialSelected);
 
          // Initial evaluation query to database
          const evalResult = await evaluateRegimenDDI(initialSelected);
          setActiveEvaluatedData(evalResult);
          setEvaluationError(null);
        }
      } catch (err) {
        console.warn("Could not load from API server, initializing default state:", err);
        setSelectedDrugs(fallbackPatient.medications.slice(0, 3));
        setLoadNotice(err.message || 'Backend unavailable. Patient data may be stale until the API is running.');
        setApiStatus('offline');
      } finally {
        setIsLoadingDB(false);
      }
    }
    loadDatabaseData();
  }, []);
 
  const handleSelectNavTab = (tabId) => {
    if (!NAV_SECTIONS.includes(tabId)) return;
    setActiveNavTab(tabId);
    const hash = `#${tabId}`;
    if (window.location.hash !== hash) {
      window.history.pushState({ section: tabId }, '', hash);
    }
    const mainEl = mainContentRef.current;
    if (mainEl) {
      mainEl.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };
 
  // Trigger polypharmacy DDI evaluation via Database API
  const handleRunEvaluation = async () => {
    const drugsSnapshot = selectedDrugs.map((d) => ({ ...d }));
    if (drugsSnapshot.length < 2) {
      return;
    }

    if (isEvaluating) return;

    if (evaluationAbortRef.current) {
      evaluationAbortRef.current.abort();
    }
    const controller = new AbortController();
    evaluationAbortRef.current = controller;

    setIsEvaluating(true);
    setEvaluationError(null);

    try {
      const evaluationResult = await evaluateRegimenDDI(drugsSnapshot, { signal: controller.signal });
      if (controller.signal.aborted) return;
      setActiveEvaluatedData(evaluationResult);
      setApiStatus('connected');
      scrollToResultsPendingRef.current = true;
    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error("Evaluation API failed:", err);
      setEvaluationError(err.message || 'Evaluation request failed.');
      setApiStatus('offline');
    } finally {
      setIsEvaluating(false);
    }
  };
 
  // Update Patient Profile in Database
  const handleUpdatePatient = async (updatedProfile) => {
    try {
      await updatePatientProfile(updatedProfile);
      const fresh = await fetchPatientProfile();
      const normalized = normalizePatientFromApi(fresh);
      if (!normalized?.name) {
        throw new Error('Patient record could not be reloaded after save.');
      }
      setPatient(normalized);
      setSelectedDrugs((prev) => {
        const next = prev.filter((m) =>
          normalized.medications.some(
            (pm) => (pm.id || pm.drugbank_id) === (m.id || m.drugbank_id)
          )
        );
        if (next.length >= 2) return next;
        return normalized.medications.slice(0, Math.min(3, normalized.medications.length));
      });
      return { ok: true, patient: normalized };
    } catch (err) {
      console.error("Failed to update patient profile in DB:", err);
      return {
        ok: false,
        message: err.message || 'Unable to reach the patient API. Check that the backend is running.',
      };
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
        setSelectedDrugs(selectedDrugs.filter((m) => m.id !== medId && m.db_id !== numericId));
      }
    } catch (err) {
      console.error("Error discontinuing medication in DB:", err);
      // Fallback local update
      const updatedMeds = patient.medications.filter((m) => m.id !== medId);
      setPatient((prev) => ({ ...prev, medications: updatedMeds }));
    }
  };
 
  const medKey = (med) => med.id || med.drugbank_id || med.name;

  // Evaluation-only selection (does not change MAR orders)
  const toggleEvalMedication = (med) => {
    setSelectedDrugs((prev) => {
      const key = medKey(med);
      const isSelected = prev.some((m) => medKey(m) === key);
      if (isSelected) {
        return prev.filter((m) => medKey(m) !== key);
      }
      return [...prev, med];
    });
  };

  const handleSelectAllEvalToggle = () => {
    setSelectedDrugs((prev) => {
      const allSelected =
        patient.medications.length > 0 &&
        patient.medications.every((med) => prev.some((m) => medKey(m) === medKey(med)));
      if (allSelected) {
        return [];
      }
      return [...patient.medications];
    });
  };

  // Dropdown Auto-Fill Handler for Add Medication Form
  const handlePresetDropdownChange = (e) => {
    const selectedId = e.target.value;
    if (!selectedId) return;
    const found = COMMON_MEDICATIONS.find((m) => m.id === selectedId);
    if (found) {
      setNewMedName(found.name);
      setNewMedId(found.id);
    }
  };

  const handleSelectDrug1 = (drugId) => {
    setSelectedDrugs((prev) => {
      if (prev.length < 2) return prev;
      const found = prev.find((m) => (m.id || m.drugbank_id) === drugId);
      if (!found) return prev;

      const others = prev.filter((m) => medKey(m) !== medKey(found));
      let second = others[0];
      if (!second) return [found];

      return [found, second, ...others.slice(1)];
    });
  };

  const handleSelectDrug2 = (drugId) => {
    setSelectedDrugs((prev) => {
      if (prev.length < 2) return prev;
      const found = prev.find((m) => (m.id || m.drugbank_id) === drugId);
      if (!found || medKey(found) === medKey(prev[0])) return prev;

      const rest = prev.filter((m) => medKey(m) !== medKey(found) && medKey(m) !== medKey(prev[0]));
      return [prev[0], found, ...rest];
    });
  };

  const handleSelectPresetPair = (pairString) => {
    const [id1, id2] = pairString.split('+');
    let d1 = patient.medications.find((m) => (m.id || m.drugbank_id) === id1);
    let d2 = patient.medications.find((m) => (m.id || m.drugbank_id) === id2);

    const medsToAdd = [];
    if (!d1) {
      const preset1 = COMMON_MEDICATIONS.find((m) => m.id === id1);
      if (preset1) medsToAdd.push({ id: preset1.id, drugbank_id: preset1.id, name: preset1.name, dosage: preset1.dosage, frequency: preset1.frequency, route: preset1.route });
    }
    if (!d2) {
      const preset2 = COMMON_MEDICATIONS.find((m) => m.id === id2);
      if (preset2) medsToAdd.push({ id: preset2.id, drugbank_id: preset2.id, name: preset2.name, dosage: preset2.dosage, frequency: preset2.frequency, route: preset2.route });
    }

    if (medsToAdd.length > 0) {
      setPatient((prev) => {
        const newMedsList = [...prev.medications, ...medsToAdd];
        d1 = newMedsList.find((m) => (m.id || m.drugbank_id) === id1);
        d2 = newMedsList.find((m) => (m.id || m.drugbank_id) === id2);
        if (d1 && d2) setSelectedDrugs([d1, d2]);
        return { ...prev, medications: newMedsList };
      });
    } else if (d1 && d2) {
      setSelectedDrugs([d1, d2]);
    }
  };

  const showWorkflowHints = activeNavTab === 'ddi';

  const evaluationMatchesSelection =
    activeEvaluatedData &&
    selectionSignature(selectedDrugs) ===
      selectionSignature(activeEvaluatedData.selectedDrugs || []);
  const resultsStale = Boolean(activeEvaluatedData && !evaluationMatchesSelection);

  const handleDownloadReport = async () => {
    setReportError(null);
    setIsGeneratingReport(true);
    try {
      let patientRecord = patient;
      try {
        const fresh = await fetchPatientProfile();
        patientRecord = normalizePatientFromApi(fresh);
        if (patientRecord?.name) {
          setPatient(patientRecord);
        }
      } catch (fetchErr) {
        if (!patientRecord?.name) {
          throw new Error(
            fetchErr.message || 'Could not load patient record. Connect to the API and try again.',
          );
        }
      }

      const evaluationForReport = evaluationMatchesSelection ? activeEvaluatedData : null;

      await generatePdfReport({
        patient: patientRecord,
        evaluation: evaluationForReport,
        currentSelection: selectedDrugs,
        engineHealth,
      });
    } catch (err) {
      console.error('PDF report generation failed:', err);
      setReportError(err.message || 'Report generation failed.');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  useEffect(() => {
    if (!scrollToResultsPendingRef.current) return;
    if (isEvaluating) return;
    if (!activeEvaluatedData || resultsStale) return;
    if (activeNavTab !== 'ddi') return;

    scrollToResultsPendingRef.current = false;
    const node = ddiResultsAnchorRef.current;
    if (!node) return;

    const t = window.setTimeout(() => {
      node.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 0);
    return () => window.clearTimeout(t);
  }, [activeEvaluatedData, isEvaluating, resultsStale, activeNavTab]);

  return (
    <div className={`app-layout screenshot-theme ${darkMode ? 'dark-theme' : ''}`}>
      {/* Top Navbar */}
      <header className="screenshot-top-nav">
        <div className="nav-brand">
          <span className="brand-dot"></span>
          <span className="brand-text">DDI Research Portal</span>
          <span className="db-status-badge">
            <span className="live-db-dot"></span>
          </span>
        </div>
 
        <div className="nav-actions">
          {/* Download Clinical PDF Report Button */}
          <button
            type="button"
            className="download-report-btn"
            onClick={handleDownloadReport}
            disabled={isGeneratingReport || isLoadingDB}
            title="Download clinical DDI evaluation report (PDF)"
            aria-busy={isGeneratingReport}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '16px', height: '16px' }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span>{isGeneratingReport ? 'Generating…' : 'Download Report'}</span>
          </button>
          {reportError && (
            <span className="report-download-error" role="alert">
              {reportError}
            </span>
          )}

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
 
        <div
          className={`api-status-pill ${apiStatus === 'connected' ? 'connected' : ''}`}
          role="status"
          aria-live="polite"
        >
          <span className="api-status-dot" aria-hidden="true" />
          <span>
            {apiStatus === 'connected'
              ? 'Model service connected'
              : apiStatus === 'checking'
                ? 'Checking connection…'
                : 'Model service offline'}
          </span>
        </div>
      </header>
 
      {/* Main 2-Column Dashboard Grid */}
      <main className="dashboard-grid-container">
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
        <section className="right-main-col" ref={mainContentRef} aria-live="polite">
          <p className="clinical-disclaimer" role="note">
            This tool shows research-model predictions and retrieved reference text. It is not a clinical
            diagnosis, prescribing guide, or substitute for a licensed clinician.
          </p>

          {showWorkflowHints && (
            <ol className="workflow-steps" aria-label="Evaluation workflow">
              <li>
                <span className="workflow-step-num">1.</span> Select two or more medications for evaluation
              </li>
              <li>
                <span className="workflow-step-num">2.</span> Run interaction evaluation
              </li>
              <li>
                <span className="workflow-step-num">3.</span> Review summary, then evidence if needed
              </li>
            </ol>
          )}

          {(loadNotice || evaluationError) && (
            <div className={`portal-alert ${evaluationError ? 'portal-alert-error' : 'portal-alert-warning'}`} role="alert">
              <strong>{evaluationError ? 'Evaluation failed' : 'Notice'}</strong>
              <span>{evaluationError || loadNotice}</span>
            </div>
          )}

          {isLoadingDB && (
            <div className="portal-loading-banner" role="status">
              Loading patient regimen and running initial DDI evaluation…
            </div>
          )}

          {activeNavTab === 'overview' && (
            <OverviewSection
              patient={patient}
              selectedDrugs={selectedDrugs}
              evaluationData={activeEvaluatedData}
              apiStatus={apiStatus}
              isLoadingDB={isLoadingDB}
              evaluationError={evaluationError}
              onGoToSection={handleSelectNavTab}
            />
          )}

          {activeNavTab === 'medications' && (
              <div className="main-panel-step" id="sec-medications">
                <div className="screenshot-card active-meds-top-card">
                  <header className="section-page-header section-page-header-pad">
                    <h2 className="section-page-title">Medications</h2>
                    <p className="section-helper">
                      View the patient&apos;s current medication orders and administration details.
                    </p>
                  </header>
                  <div className="mar-card-header">
                    <div className="header-left">
                      <div className="section-title-with-badge">
                        <h3 className="card-main-title">Active Regimen & MAR Table</h3>
                        <span className="med-count-pill">
                          {patient.medications.length} / {MAX_MEDICATIONS} Max (Database Capacity)
                        </span>
                      </div>
                    </div>
 
                    <div className="mar-header-actions">
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
                      <select 
                        className="add-med-preset-select"
                        defaultValue=""
                        onChange={handlePresetDropdownChange}
                      >
                        <option value="" disabled>-- Choose from Clinical Drug Database --</option>
                        {COMMON_MEDICATIONS.map((med) => (
                          <option key={med.id} value={med.id}>
                            {med.name} ({med.id}) &bull; {med.category}
                          </option>
                        ))}
                      </select>

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
                          <th>DRUGBANK ID</th>
                          <th>MEDICATION NAME</th>
                          <th>DOSAGE</th>
                          <th>FREQUENCY</th>
                          <th>ROUTE</th>
                          <th>ACTION</th>
                        </tr>
                      </thead>
                      <tbody>
                        {patient.medications.map((med) => (
                            <tr key={med.id || med.drugbank_id}>
                              <td className="mono-code">{med.id || med.drugbank_id}</td>
                              <td className="drug-name-col">
                                <strong>{med.name}</strong>
                              </td>
                              <td>{med.dosage || '—'}</td>
                              <td>{med.frequency || '—'}</td>
                              <td>{med.route || '—'}</td>
                              <td className="right-action">
                                {patient.medications.length > 2 && (
                                  <button
                                    type="button"
                                    className="discontinue-btn"
                                    onClick={() => handleRemoveMedication(med.id)}
                                  >
                                    Discontinue
                                  </button>
                                )}
                              </td>
                            </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
          )}

          {activeNavTab === 'ddi' && (
            <>
              <div className="main-panel-step" id="sec-pair">
                <header className="section-page-header section-page-header-standalone">
                  <h2 className="section-page-title">Interaction results</h2>
                  <p className="section-helper">
                    Select two or more medications to evaluate potential drug–drug interactions.
                  </p>
                </header>
                <DrugPairCard
                  selectedDrugs={selectedDrugs}
                  availableMeds={patient.medications}
                  evaluationData={activeEvaluatedData}
                  onRunEvaluation={handleRunEvaluation}
                  isEvaluating={isEvaluating}
                  onSelectDrug1={handleSelectDrug1}
                  onSelectDrug2={handleSelectDrug2}
                  onSelectPresetPair={handleSelectPresetPair}
                  onToggleEvalMedication={toggleEvalMedication}
                  onSelectAllEval={handleSelectAllEvalToggle}
                  resultsStale={resultsStale}
                />
              </div>

              <div className="main-panel-step ddi-results-block" id="sec-prediction">
                {isEvaluating && (
                  <div className="portal-loading-banner" role="status">
                    Running T-GNN + RAG evaluation across {selectedDrugs.length} medications…
                  </div>
                )}
                <DdiEvaluationResults
                  evaluationData={activeEvaluatedData}
                  isEvaluating={isEvaluating}
                  resultsStale={resultsStale}
                  resultsAnchorRef={ddiResultsAnchorRef}
                />
              </div>
            </>
          )}

          {activeNavTab === 'rag' && (
            <div className="main-panel-step" id="sec-rag">
              <header className="section-page-header section-page-header-standalone">
                <h2 className="section-page-title">Evidence &amp; explanation</h2>
                <p className="section-helper">
                  Retrieved pharmacology context (RAG) for the latest evaluation. Expand sections for detail.
                </p>
              </header>
              {!activeEvaluatedData ? (
                <div className="screenshot-card portal-empty-state">
                  <p className="portal-empty-title">No explanation available</p>
                  <p className="portal-empty-copy">Run an interaction evaluation to populate RAG text.</p>
                </div>
              ) : (
                <>
                  <div className="screenshot-card rag-context-banner">
                    <div className="banner-left">
                      <RagIcon className="banner-icon-svg" />
                      <div className="banner-text">
                        Regimen: {selectedDrugs.map((d) => d.name).join(' · ')}
                      </div>
                    </div>
                    <span
                      className={`banner-sev-pill ${activeEvaluatedData?.prediction?.severity?.toLowerCase()}`}
                    >
                      {activeEvaluatedData?.prediction?.severity} ·{' '}
                      {Math.round((activeEvaluatedData?.prediction?.probability || 0) * 100)}% top score
                    </span>
                  </div>
                  <div className="screenshot-card history-section-card rag-only-card">
                    <div className="history-timeline-container rag-evidence-stack">
                      <div className="timeline-vertical-line" />
                      <RAGExplanation ragData={activeEvaluatedData.rag_explanation} />
                      <EvidenceTree
                        evidence={activeEvaluatedData.evidence}
                        selectedDrugs={selectedDrugs}
                      />
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {activeNavTab === 'evidence' && (
            <SourceAuditPanel
              evaluationData={activeEvaluatedData}
              engineHealth={engineHealth}
              apiStatus={apiStatus}
            />
          )}

        </section>
      </main>
    </div>
  );
}
 