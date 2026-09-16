import React, { useState } from 'react';
import './PatientCard.css';

export function PatientCard({ 
  patient, 
  onUpdatePatient, 
  activeNavTab, 
  onSelectNavTab 
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(patient.name);
  const [editAge, setEditAge] = useState(patient.age);
  const [editGender, setEditGender] = useState(patient.gender);

  const [generalOpen, setGeneralOpen] = useState(true);
  const [contactOpen, setContactOpen] = useState(true);

  const handleSaveProfile = (e) => {
    e.preventDefault();
    if (!editName.trim()) return;
    onUpdatePatient({
      name: editName.trim(),
      age: parseInt(editAge) || patient.age,
      gender: editGender
    });
    setIsEditing(false);
  };

  const navItems = [
    { id: 'overview', label: 'Overview', icon: '👤' },
    { id: 'medications', label: `Active Medications (${patient.medications.length})`, icon: '💊' },
    { id: 'ddi', label: 'DDI Risk Prediction', icon: '⚠️' },
    { id: 'rag', label: 'RAG Explanation', icon: '🔬' },
    { id: 'evidence', label: 'Evidence Audit', icon: '📚' },
  ];

  return (
    <div className="screenshot-card patient-sidebar-card">
      {/* Header Avatar & Basic Info */}
      <div className="sidebar-patient-header">
        <div className="patient-purple-avatar">
          {patient.name ? patient.name.charAt(0).toUpperCase() : 'E'}
        </div>

        {!isEditing ? (
          <div className="sidebar-name-block">
            <h2 className="patient-sidebar-name">{patient.name}</h2>
            <span className="patient-sub-demo">{patient.gender}, {patient.age}</span>
          </div>
        ) : (
          <form className="sidebar-edit-form" onSubmit={handleSaveProfile}>
            <input 
              type="text" 
              value={editName} 
              onChange={(e) => setEditName(e.target.value)} 
              placeholder="Name"
              required
            />
            <div className="sidebar-edit-row">
              <input 
                type="number" 
                value={editAge} 
                onChange={(e) => setEditAge(e.target.value)} 
                placeholder="Age"
                className="age-field"
                required
              />
              <select value={editGender} onChange={(e) => setEditGender(e.target.value)}>
                <option value="Female">Female</option>
                <option value="Male">Male</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div className="sidebar-edit-btns">
              <button type="submit" className="save-mini-btn">Save</button>
              <button type="button" className="cancel-mini-btn" onClick={() => setIsEditing(false)}>Cancel</button>
            </div>
          </form>
        )}

        {/* Action Button Row */}
        <div className="sidebar-action-bar">
          <button 
            className="chart-primary-btn" 
            onClick={() => setIsEditing(!isEditing)}
            title="Edit Patient Profile"
          >
            <svg className="chart-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span>{isEditing ? 'Close Edit' : 'Open Patient Chart'}</span>
          </button>
          
          <button className="icon-square-btn" title="Call Patient">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M6.62 10.79c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.27-.27.67-.36 1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2z"/>
            </svg>
          </button>

          <button className="icon-square-btn" title="Schedule Appointment">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 4h-1V2h-2v2H8V2H6v2H5c-1.11 0-1.99.9-1.99 2L3 20c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 16H5V10h14v10zm0-12H5V6h14v2z"/>
            </svg>
          </button>

          <button className="icon-square-btn" title="More Options">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M6 10c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm12 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm-6 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Navigation Links Menu */}
      <div className="sidebar-nav-menu">
        {navItems.map((item) => (
          <div 
            key={item.id}
            className={`nav-item-link ${activeNavTab === item.id ? 'active' : ''}`}
            onClick={() => onSelectNavTab(item.id)}
          >
            <div className="nav-item-left">
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </div>
            <svg className="chevron-right" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
          </div>
        ))}
      </div>

      {/* General Info Collapsible Section */}
      <div className="sidebar-info-section">
        <div 
          className="section-collapsible-header"
          onClick={() => setGeneralOpen(!generalOpen)}
        >
          <span className="section-title-text">General Info</span>
          <svg className={`chevron-down ${generalOpen ? 'open' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>

        {generalOpen && (
          <div className="info-kv-list">
            <div className="kv-row">
              <span className="kv-key">Gender</span>
              <span className="kv-val">{patient.gender}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">DOB</span>
              <span className="kv-val">{patient.dob || '1954-07-22'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Preferred Language</span>
              <span className="kv-val">{patient.language || 'English'}</span>
            </div>
          </div>
        )}
      </div>

      {/* Contact Info Collapsible Section */}
      <div className="sidebar-info-section">
        <div 
          className="section-collapsible-header"
          onClick={() => setContactOpen(!contactOpen)}
        >
          <span className="section-title-text">Contact Info</span>
          <svg className={`chevron-down ${contactOpen ? 'open' : ''}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>

        {contactOpen && (
          <div className="info-kv-list">
            <div className="kv-row">
              <span className="kv-key">Phone</span>
              <span className="kv-val">{patient.phone || '(406) 555-0120'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Emergency Contact</span>
              <span className="kv-val">{patient.emergencyContact || '(480) 555-0103'}</span>
            </div>
            <div className="kv-row">
              <span className="kv-key">Email Address</span>
              <span className="kv-val link">{patient.email || 'eleanor.vance@example.com'}</span>
            </div>
            <div className="kv-row address-row">
              <span className="kv-key">Mailing Address</span>
              <span className="kv-val right-align">{patient.address || '2972 Westheimer Rd. Santa Ana, Illinois 85486'}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
