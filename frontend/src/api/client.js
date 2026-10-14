const API_BASE_URL = '/api';

/**
 * API Client Layer for T-GNN Clinical Portal
 * Connects frontend directly to Python Flask REST API & SQLite Database
 */

async function parseApiError(response) {
  let message = `Request failed (${response.status})`;
  try {
    const payload = await response.json();
    if (payload?.error) {
      message = payload.detail ? `${payload.error}: ${payload.detail}` : payload.error;
    }
  } catch {
    // ignore JSON parse errors
  }
  return message;
}

export async function fetchPatientProfile() {
  const response = await fetch(`${API_BASE_URL}/patient`);
  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }
  return response.json();
}

export async function updatePatientProfile(profileData) {
  const response = await fetch(`${API_BASE_URL}/patient`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profileData),
  });
  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }
  return response.json();
}

export async function addMedicationOrder(medData) {
  const response = await fetch(`${API_BASE_URL}/medications`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(medData),
  });
  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }
  return response.json();
}

export async function discontinueMedicationOrder(medId) {
  const response = await fetch(`${API_BASE_URL}/medications/${medId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }
  return response.json();
}

export async function evaluateRegimenDDI(selectedDrugs, options = {}) {
  const response = await fetch(`${API_BASE_URL}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ selectedDrugs }),
    signal: options.signal,
  });
  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }
  return response.json();
}

export async function fetchDrugCatalog() {
  try {
    const response = await fetch(`${API_BASE_URL}/catalog`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (err) {
    console.error('API Error fetching drug catalog:', err);
    return [];
  }
}

export async function fetchHealthStatus() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }
  return response.json();
}
