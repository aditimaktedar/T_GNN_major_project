import React, { useState, useEffect, useCallback } from 'react';
import './PatientCard.css';
import { DobDatePicker } from './DobDatePicker';
import { BookIcon, ChartIcon, MedicationsIcon, RagIcon, UserNavIcon } from './SegmentIcons';

const LANGUAGE_OPTIONS = ['English', 'Spanish', 'French', 'Mandarin', 'Hindi', 'Other'];

function parseWeightKg(weightStr) {
  if (weightStr == null || weightStr === '') return '';
  const match = String(weightStr).match(/([\d.]+)/);
  return match ? match[1] : '';
}

function calculateAgeFromDob(dobStr) {
  if (!dobStr) return null;
  const dob = new Date(`${dobStr}T12:00:00`);
  if (Number.isNaN(dob.getTime())) return null;
  const today = new Date();
  let age = today.getFullYear() - dob.getFullYear();
  const monthDiff = today.getMonth() - dob.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < dob.getDate())) {
    age -= 1;
  }
  return age;
}

function validateProfile(fields) {
  const errors = {};
  const trimmed = (fields.name || '').trim();
  if (!trimmed) {
    errors.name = 'Full name is required.';
  }
  if (!fields.gender) {
    errors.gender = 'Select a gender.';
  }

  if (fields.dob) {
    const dobDate = new Date(`${fields.dob}T12:00:00`);
    if (Number.isNaN(dobDate.getTime())) {
      errors.dob = 'Enter a valid date of birth.';
    } else {
      const today = new Date();
      today.setHours(23, 59, 59, 999);
      if (dobDate > today) {
        errors.dob = 'Date of birth cannot be in the future.';
      }
    }
  }

  if (!fields.dob) {
    const ageNum = parseInt(String(fields.age), 10);
    if (Number.isNaN(ageNum) || ageNum < 1 || ageNum > 120) {
      errors.age = 'Enter an age between 1 and 120.';
    }
  }

  if (fields.weightKg !== '' && fields.weightKg != null) {
    const w = parseFloat(String(fields.weightKg));
    if (Number.isNaN(w) || w <= 0 || w > 500) {
      errors.weightKg = 'Enter weight between 0 and 500 kg.';
    }
  }

  return errors;
}

export function PatientCard({
  patient,
  onUpdatePatient,
  activeNavTab,
  onSelectNavTab,
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(patient.name);
  const [editAge, setEditAge] = useState(patient.age);
  const [editGender, setEditGender] = useState(patient.gender);
  const [editDob, setEditDob] = useState(patient.dob || '');
  const [editLanguage, setEditLanguage] = useState(patient.language || '');
  const [editWeightKg, setEditWeightKg] = useState(parseWeightKg(patient.weight));
  const [editRoom, setEditRoom] = useState(patient.room || '');
  const [editPhysician, setEditPhysician] = useState(patient.physician || '');
  const [fieldErrors, setFieldErrors] = useState({});
  const [saveStatus, setSaveStatus] = useState(null);
  const [saveMessage, setSaveMessage] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const [generalOpen, setGeneralOpen] = useState(true);

  const resetEditFromPatient = useCallback(() => {
    setEditName(patient.name);
    setEditAge(patient.age);
    setEditGender(patient.gender);
    setEditDob(patient.dob || '');
    setEditLanguage(patient.language || '');
    setEditWeightKg(parseWeightKg(patient.weight));
    setEditRoom(patient.room || '');
    setEditPhysician(patient.physician || '');
    setFieldErrors({});
  }, [patient]);

  const isDirty =
    isEditing &&
    (editName.trim() !== (patient.name || '').trim() ||
      String(editAge) !== String(patient.age ?? '') ||
      editGender !== patient.gender ||
      (editDob || '') !== (patient.dob || '') ||
      (editLanguage || '') !== (patient.language || '') ||
      String(editWeightKg) !== String(parseWeightKg(patient.weight)) ||
      (editRoom || '') !== (patient.room || '') ||
      (editPhysician || '') !== (patient.physician || ''));

  useEffect(() => {
    const onBeforeUnload = (e) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [isDirty]);

  const handleDobChange = (value) => {
    setEditDob(value);
    if (value) {
      const derived = calculateAgeFromDob(value);
      if (derived != null && derived >= 1 && derived <= 120) {
        setEditAge(derived);
      }
    }
  };

  const handleStartEdit = () => {
    resetEditFromPatient();
    setSaveStatus(null);
    setSaveMessage('');
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    if (isDirty && !window.confirm('Discard unsaved changes to this profile?')) {
      return;
    }
    resetEditFromPatient();
    setIsEditing(false);
    setSaveStatus(null);
    setSaveMessage('');
  };

  const handleSaveProfile = async (e) => {
    e.preventDefault();
    const errors = validateProfile({
      name: editName,
      age: editAge,
      gender: editGender,
      dob: editDob,
      weightKg: editWeightKg,
    });
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) return;

    const dobTrimmed = editDob?.trim() || null;
    let ageToSave = parseInt(String(editAge), 10);
    if (dobTrimmed) {
      ageToSave = calculateAgeFromDob(dobTrimmed);
      if (ageToSave == null) {
        setFieldErrors({ dob: 'Enter a valid date of birth.' });
        return;
      }
    }

    setIsSaving(true);
    setSaveStatus(null);
    setSaveMessage('');

    const payload = {
      name: editName.trim(),
      age: ageToSave,
      gender: editGender,
      dob: dobTrimmed,
      language: editLanguage.trim() || null,
      room: editRoom.trim() || null,
      physician: editPhysician.trim() || null,
    };

    if (editWeightKg !== '' && editWeightKg != null) {
      payload.weight_kg = parseFloat(String(editWeightKg));
    }

    try {
      const result = await onUpdatePatient(payload);
      if (!result || result.ok !== true) {
        setSaveStatus('error');
        setSaveMessage(result?.message || 'Could not save profile. Try again when the API is available.');
        return;
      }
      setSaveStatus('success');
      setSaveMessage('Profile saved.');
      setIsEditing(false);
      setFieldErrors({});
    } catch {
      setSaveStatus('error');
      setSaveMessage('Could not save profile.');
    } finally {
      setIsSaving(false);
    }
  };

  const navItems = [
    { id: 'overview', label: 'Overview', Icon: UserNavIcon },
    { id: 'medications', label: `Medications (${patient.medications.length})`, Icon: MedicationsIcon },
    { id: 'ddi', label: 'Interaction results', Icon: ChartIcon },
    { id: 'rag', label: 'Evidence & explanation', Icon: RagIcon },
    { id: 'evidence', label: 'Source audit', Icon: BookIcon },
  ];

  const languageOptions =
    patient.language && !LANGUAGE_OPTIONS.includes(patient.language)
      ? [patient.language, ...LANGUAGE_OPTIONS]
      : LANGUAGE_OPTIONS;

  const initials =
    patient.initials ||
    (patient.name
      ? patient.name
          .split(/\s+/)
          .map((p) => p[0])
          .join('')
          .slice(0, 2)
          .toUpperCase()
      : '?');

  return (
    <div className="screenshot-card patient-sidebar-card">
      <div className="sidebar-patient-header">
        <div className="patient-accent-avatar" aria-hidden="true">
          {initials}
        </div>

        {!isEditing ? (
          <div className="sidebar-name-block">
            <h2 className="patient-sidebar-name">{patient.name}</h2>
            <span className="patient-sub-demo">
              {patient.gender}, {patient.age} years
            </span>
            {patient.mrn && <span className="patient-mrn-label">MRN {patient.mrn}</span>}
          </div>
        ) : (
          <form className="sidebar-edit-form sidebar-edit-form-scroll" onSubmit={handleSaveProfile} noValidate>
            <p className="edit-form-hint">Changes are saved to the clinical database. Age is derived from date of birth when DOB is provided.</p>

            <p className="edit-section-heading">Personal information</p>
            <label className="field-label" htmlFor="patient-edit-name">
              Full name
            </label>
            <input
              id="patient-edit-name"
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              className={fieldErrors.name ? 'input-invalid' : ''}
            />
            {fieldErrors.name && <span className="field-error">{fieldErrors.name}</span>}

            <span className="field-label" id="patient-edit-dob-label">
              Date of birth
            </span>
            <DobDatePicker
              key={`dob-picker-${patient.dob ?? 'none'}`}
              id="patient-edit-dob"
              value={editDob}
              onChange={handleDobChange}
              invalid={Boolean(fieldErrors.dob)}
            />
            {fieldErrors.dob && <span className="field-error">{fieldErrors.dob}</span>}

            <div className="sidebar-edit-row">
              <div className="field-stack">
                <label className="field-label" htmlFor="patient-edit-age">
                  Age (years)
                </label>
                <input
                  id="patient-edit-age"
                  type="number"
                  min={1}
                  max={120}
                  value={editAge}
                  onChange={(e) => setEditAge(e.target.value)}
                  disabled={Boolean(editDob)}
                  className={`age-field ${fieldErrors.age ? 'input-invalid' : ''}`}
                />
                {editDob && (
                  <span className="field-hint">Calculated from date of birth</span>
                )}
                {fieldErrors.age && <span className="field-error">{fieldErrors.age}</span>}
              </div>
              <div className="field-stack">
                <label className="field-label" htmlFor="patient-edit-gender">
                  Gender
                </label>
                <select
                  id="patient-edit-gender"
                  value={editGender}
                  onChange={(e) => setEditGender(e.target.value)}
                  className={fieldErrors.gender ? 'input-invalid' : ''}
                >
                  <option value="Female">Female</option>
                  <option value="Male">Male</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>

            <label className="field-label" htmlFor="patient-edit-language">
              Preferred language
            </label>
            <select
              id="patient-edit-language"
              value={editLanguage}
              onChange={(e) => setEditLanguage(e.target.value)}
            >
              <option value="">—</option>
              {languageOptions.map((lang) => (
                <option key={lang} value={lang}>
                  {lang}
                </option>
              ))}
            </select>

            <p className="edit-section-heading">Clinical information</p>
            <label className="field-label" htmlFor="patient-edit-weight">
              Weight (kg)
            </label>
            <input
              id="patient-edit-weight"
              type="number"
              min={0.1}
              max={500}
              step={0.1}
              value={editWeightKg}
              onChange={(e) => setEditWeightKg(e.target.value)}
              className={fieldErrors.weightKg ? 'input-invalid' : ''}
              placeholder="e.g. 68"
            />
            {fieldErrors.weightKg && <span className="field-error">{fieldErrors.weightKg}</span>}

            <label className="field-label" htmlFor="patient-edit-room">
              Care location
            </label>
            <input
              id="patient-edit-room"
              type="text"
              value={editRoom}
              onChange={(e) => setEditRoom(e.target.value)}
            />

            <label className="field-label" htmlFor="patient-edit-physician">
              Attending physician
            </label>
            <input
              id="patient-edit-physician"
              type="text"
              value={editPhysician}
              onChange={(e) => setEditPhysician(e.target.value)}
            />

            <div className="sidebar-edit-btns">
              <button type="submit" className="save-mini-btn" disabled={isSaving}>
                {isSaving ? 'Saving…' : 'Save'}
              </button>
              <button type="button" className="cancel-mini-btn" onClick={handleCancelEdit} disabled={isSaving}>
                Cancel
              </button>
            </div>
          </form>
        )}

        {saveStatus && !isEditing && (
          <div className={`profile-save-banner ${saveStatus === 'success' ? 'success' : 'error'}`} role="status">
            {saveMessage}
          </div>
        )}

        <div className="sidebar-action-bar">
          {!isEditing ? (
            <button type="button" className="chart-primary-btn" onClick={handleStartEdit} title="Edit patient profile">
              <svg className="chart-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                />
              </svg>
              <span>Edit profile</span>
            </button>
          ) : (
            <button type="button" className="chart-primary-btn secondary" onClick={handleCancelEdit}>
              <span>Close editor</span>
            </button>
          )}
        </div>
      </div>

      <div className="sidebar-nav-menu">
        {navItems.map((item) => (
          <button
            type="button"
            key={item.id}
            className={`nav-item-link ${activeNavTab === item.id ? 'active' : ''}`}
            onClick={() => onSelectNavTab(item.id)}
          >
            <div className="nav-item-left">
              <span className="nav-icon">
                <item.Icon className="nav-icon-svg" />
              </span>
              <span className="nav-label">{item.label}</span>
            </div>
            <svg className="chevron-right" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
          </button>
        ))}
      </div>

      <div className="sidebar-info-section">
        <button
          type="button"
          className="section-collapsible-header"
          onClick={() => setGeneralOpen(!generalOpen)}
          aria-expanded={generalOpen}
        >
          <span className="section-title-text">General information</span>
          <svg className={`chevron-down ${generalOpen ? 'open' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {generalOpen && (
          <div className="info-kv-list">
            <div className="kv-row">
              <span className="kv-key">Gender</span>
              <span className="kv-val">{patient.gender || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Date of birth</span>
              <span className="kv-val">{patient.dob || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Preferred language</span>
              <span className="kv-val">{patient.language || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Weight</span>
              <span className="kv-val">{patient.weight || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Care location</span>
              <span className="kv-val">{patient.room || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Attending physician</span>
              <span className="kv-val">{patient.physician || '—'}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
