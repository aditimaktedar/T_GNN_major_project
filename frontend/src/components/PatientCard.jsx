import React, { useState, useEffect, useCallback } from 'react';
import './PatientCard.css';
import { BookIcon, ChartIcon, MedicationsIcon, RagIcon, UserNavIcon } from './SegmentIcons';

function validateProfile({ name, age, gender }) {
  const errors = {};
  const trimmed = (name || '').trim();
  if (!trimmed) {
    errors.name = 'Full name is required.';
  }
  const ageNum = parseInt(String(age), 10);
  if (Number.isNaN(ageNum) || ageNum < 1 || ageNum > 120) {
    errors.age = 'Enter an age between 1 and 120.';
  }
  if (!gender) {
    errors.gender = 'Select a gender.';
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
  const [fieldErrors, setFieldErrors] = useState({});
  const [saveStatus, setSaveStatus] = useState(null); // 'success' | 'error' | null
  const [saveMessage, setSaveMessage] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const [generalOpen, setGeneralOpen] = useState(true);
  const [contactOpen, setContactOpen] = useState(false);

  const resetEditFromPatient = useCallback(() => {
    setEditName(patient.name);
    setEditAge(patient.age);
    setEditGender(patient.gender);
    setFieldErrors({});
  }, [patient.name, patient.age, patient.gender]);

  const isDirty =
    isEditing &&
    (editName.trim() !== (patient.name || '').trim() ||
      String(editAge) !== String(patient.age) ||
      editGender !== patient.gender);

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
    const errors = validateProfile({ name: editName, age: editAge, gender: editGender });
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) return;

    setIsSaving(true);
    setSaveStatus(null);
    setSaveMessage('');

    const payload = {
      name: editName.trim(),
      age: parseInt(String(editAge), 10),
      gender: editGender,
    };

    try {
      const result = await onUpdatePatient(payload);
      if (result?.ok === false) {
        setSaveStatus('error');
        setSaveMessage(result.message || 'Could not save profile. Try again when the API is available.');
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
            {patient.mrn && (
              <span className="patient-mrn-label">MRN {patient.mrn}</span>
            )}
          </div>
        ) : (
          <form className="sidebar-edit-form" onSubmit={handleSaveProfile} noValidate>
            <p className="edit-form-hint">Editable fields sync to the clinical database when saved.</p>
            <label className="field-label" htmlFor="patient-edit-name">
              Full name
            </label>
            <input
              id="patient-edit-name"
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              className={fieldErrors.name ? 'input-invalid' : ''}
              aria-invalid={!!fieldErrors.name}
              aria-describedby={fieldErrors.name ? 'err-name' : undefined}
            />
            {fieldErrors.name && (
              <span id="err-name" className="field-error" role="alert">
                {fieldErrors.name}
              </span>
            )}

            <div className="sidebar-edit-row">
              <div className="field-stack">
                <label className="field-label" htmlFor="patient-edit-age">
                  Age
                </label>
                <input
                  id="patient-edit-age"
                  type="number"
                  min={1}
                  max={120}
                  value={editAge}
                  onChange={(e) => setEditAge(e.target.value)}
                  className={`age-field ${fieldErrors.age ? 'input-invalid' : ''}`}
                  aria-invalid={!!fieldErrors.age}
                />
                {fieldErrors.age && (
                  <span className="field-error" role="alert">
                    {fieldErrors.age}
                  </span>
                )}
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
          <div
            className={`profile-save-banner ${saveStatus === 'success' ? 'success' : 'error'}`}
            role="status"
          >
            {saveMessage}
          </div>
        )}

        <div className="sidebar-action-bar">
          {!isEditing ? (
            <button
              type="button"
              className="chart-primary-btn"
              onClick={handleStartEdit}
              title="Edit name, age, and gender"
            >
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
              <span className="kv-val">{patient.gender}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Date of birth</span>
              <span className="kv-val read-only-val">{patient.dob || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Preferred language</span>
              <span className="kv-val read-only-val">{patient.language || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Weight</span>
              <span className="kv-val read-only-val">{patient.weight || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Care location</span>
              <span className="kv-val read-only-val">{patient.room || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Attending physician</span>
              <span className="kv-val read-only-val">{patient.physician || '—'}</span>
            </div>
          </div>
        )}
      </div>

      <div className="sidebar-info-section">
        <button
          type="button"
          className="section-collapsible-header"
          onClick={() => setContactOpen(!contactOpen)}
          aria-expanded={contactOpen}
        >
          <span className="section-title-text">Contact (read-only)</span>
          <svg className={`chevron-down ${contactOpen ? 'open' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {contactOpen && (
          <div className="info-kv-list">
            <div className="kv-row">
              <span className="kv-key">Phone</span>
              <span className="kv-val read-only-val">{patient.phone || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Emergency contact</span>
              <span className="kv-val read-only-val">{patient.emergencyContact || '—'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Email</span>
              <span className="kv-val read-only-val">{patient.email || '—'}</span>
            </div>
            <div className="kv-row address-row">
              <span className="kv-key">Address</span>
              <span className="kv-val read-only-val right-align">{patient.address || '—'}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
