"""Patient profile API persistence."""

from datetime import date, timedelta

from src.server import app, _calculate_age_from_dob


def test_calculate_age_from_dob():
    today = date.today()
    dob = today.replace(year=today.year - 30)
    assert _calculate_age_from_dob(dob.isoformat()) == 30


def test_put_patient_extended_fields():
    client = app.test_client()
    payload = {
        "name": "Eleanor Vance",
        "gender": "Female",
        "dob": "1954-07-22",
        "language": "English",
        "weight_kg": 68,
        "room": "Bed 304-B (Step-down Unit)",
        "physician": "Dr. Jordan Hughes",
    }
    r = client.put("/api/patient", json=payload)
    assert r.status_code == 200
    body = r.get_json()
    assert body["language"] == "English"
    assert body["weight"] == "68 kg"
    assert body["room"] == payload["room"]
    assert body["physician"] == payload["physician"]
    assert body["dob"] == payload["dob"]
    assert body["age"] == _calculate_age_from_dob(payload["dob"])


def test_put_patient_partial_update_preserves_other_fields():
    client = app.test_client()
    base = {
        "name": "Eleanor Vance",
        "gender": "Female",
        "dob": "1954-07-22",
        "language": "English",
        "weight_kg": 68,
        "room": "Bed 304-B",
        "physician": "Dr. Jordan Hughes",
    }
    client.put("/api/patient", json=base)
    r = client.put("/api/patient", json={"name": "Eleanor Vance Updated", "gender": "Female"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["name"] == "Eleanor Vance Updated"
    assert body["language"] == "English"
    assert body["room"] == "Bed 304-B"


def test_get_patient_age_matches_dob():
    client = app.test_client()
    dob = "1954-07-22"
    client.put(
        "/api/patient",
        json={"name": "Age Sync Test", "gender": "Female", "dob": dob},
    )
    r = client.get("/api/patient")
    assert r.status_code == 200
    body = r.get_json()
    assert body["dob"] == dob
    assert body["age"] == _calculate_age_from_dob(dob)


def test_put_patient_rejects_future_dob():
    client = app.test_client()
    future = (date.today() + timedelta(days=1)).isoformat()
    r = client.put(
        "/api/patient",
        json={"name": "Test User", "gender": "Other", "dob": future, "age": 30},
    )
    assert r.status_code == 400
