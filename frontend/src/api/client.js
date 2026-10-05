const API_BASE_URL = '/api';

/**
 * API Client Layer for T-GNN Clinical Portal
 * Connects frontend directly to Python Flask REST API & SQLite Database
 */

export async function fetchPatientProfile() {
  try {
    const response = await fetch(`${API_BASE_URL}/patient`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (err) {
    console.error("API Error fetching patient profile:", err);
    throw err;
  }
}

export async function updatePatientProfile(profileData) {
  try {
    const response = await fetch(`${API_BASE_URL}/patient`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profileData),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (err) {
    console.error("API Error updating patient profile:", err);
    throw err;
  }
}

export async function addMedicationOrder(medData) {
  try {
    const response = await fetch(`${API_BASE_URL}/medications`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(medData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.error("API Error adding medication order:", err);
    throw err;
  }
}

export async function discontinueMedicationOrder(medId) {
  try {
    const response = await fetch(`${API_BASE_URL}/medications/${medId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.error("API Error discontinuing medication order:", err);
    throw err;
  }
}

export async function evaluateRegimenDDI(selectedDrugs) {
  try {
    const response = await fetch(`${API_BASE_URL}/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selectedDrugs }),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (err) {
    console.error("API Error evaluating regimen DDI:", err);
    throw err;
  }
}

export async function fetchDrugCatalog() {
  try {
    const response = await fetch(`${API_BASE_URL}/catalog`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (err) {
    console.error("API Error fetching drug catalog:", err);
    return [];
  }
}
