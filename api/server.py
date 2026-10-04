"""
HTTP API Server exposing the merged T-GNN and RAG clinical evaluation service.
Supports both FastAPI (if installed) and standard library HTTP server fallback
with full CORS support for the React frontend.
"""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

from api.service import ClinicalDDIOrchestrator


class DDIRequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        orch = ClinicalDDIOrchestrator.get_instance()

        if path in ("/health", "/api/health"):
            self._send_json(200, orch.health())

        elif path == "/api/drugs":
            q = params.get("q", [None])[0]
            limit = int(params.get("limit", [50])[0])
            drugs = orch.resolver.list_all_drugs(query=q, limit=limit)
            self._send_json(200, {"drugs": drugs, "count": len(drugs)})

        elif path == "/api/gnn/predict":
            drug_a = params.get("drug_a", [None])[0]
            drug_b = params.get("drug_b", [None])[0]
            fold = int(params.get("fold", [1])[0]) if "fold" in params else None
            if not drug_a or not drug_b:
                self._send_json(400, {"error": "Missing required query parameters: drug_a and drug_b"})
                return
            entity_a = orch.resolver.resolve(drug_a)
            entity_b = orch.resolver.resolve(drug_b)
            pred = orch.gnn.predict(entity_a, entity_b, fold=fold)
            self._send_json(200, pred.to_dict())

        elif path == "/api/rag/evidence":
            drug_a = params.get("drug_a", [None])[0]
            drug_b = params.get("drug_b", [None])[0]
            top_k = int(params.get("top_k", [5])[0])
            if not drug_a or not drug_b:
                self._send_json(400, {"error": "Missing required query parameters: drug_a and drug_b"})
                return
            entity_a = orch.resolver.resolve(drug_a)
            entity_b = orch.resolver.resolve(drug_b)
            ev = orch.rag.retrieve_evidence(entity_a, entity_b, top_k=top_k)
            self._send_json(200, ev)

        else:
            self._send_json(404, {"error": f"Endpoint not found: {path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            payload = json.loads(post_body) if post_body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        orch = ClinicalDDIOrchestrator.get_instance()

        if path in ("/api/evaluate", "/api/predict"):
            drug_a = payload.get("drug_a") or payload.get("drug1") or payload.get("drug_a_id")
            drug_b = payload.get("drug_b") or payload.get("drug2") or payload.get("drug_b_id")

            # Fallback to names if IDs are absent
            if not drug_a and "drug_a_name" in payload:
                drug_a = payload["drug_a_name"]
            if not drug_b and "drug_b_name" in payload:
                drug_b = payload["drug_b_name"]

            if not drug_a or not drug_b:
                self._send_json(400, {
                    "error": "Both drug_a and drug_b must be specified (names or DrugBank IDs)."
                })
                return

            fold = payload.get("fold")
            top_k = int(payload.get("top_k", 5))
            patient_id = payload.get("patient_id")

            try:
                res = orch.evaluate_pair(
                    drug_a_query=drug_a,
                    drug_b_query=drug_b,
                    fold=fold,
                    top_k=top_k,
                    patient_id=patient_id
                )
                self._send_json(200, res.to_dict())
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif path == "/api/evaluate-regimen":
            medications = payload.get("medications", [])
            patient_id = payload.get("patient_id")
            fold = payload.get("fold")

            if not medications or not isinstance(medications, list):
                self._send_json(400, {"error": "Request body must contain 'medications' list."})
                return

            try:
                res = orch.evaluate_regimen(
                    medications=medications,
                    fold=fold,
                    patient_id=patient_id
                )
                self._send_json(200, res)
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        else:
            self._send_json(404, {"error": f"Endpoint not found: {path}"})

    def log_message(self, format, *args):
        # Concise logging to stdout
        sys.stdout.write(f"[API] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    server = HTTPServer((host, port), DDIRequestHandler)
    print(f"============================================================")
    print(f"TemporalDDI-GNN + RAG Merged API Server")
    print(f"Listening on http://{host}:{port}")
    print(f"Health check: http://{host}:{port}/api/health")
    print(f"============================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port=port)
