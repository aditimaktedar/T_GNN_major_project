# TemporalDDI-GNN: Temporal Graph Neural Network & RAG Clinical Portal

A clinical decision support portal powered by **Temporal Graph Neural Networks (T-GNN)** and **Retrieval-Augmented Generation (RAG)** for multi-drug polypharmacy Drug-Drug Interaction (DDI) risk evaluation and evidence auditing.

---

## 🌟 Key Features

### 1. 💊 Polypharmacy & Multi-Drug DDI Evaluation (3+ Drugs Supported)
- **High-Order Regimen Evaluation**: Test 2, 3, 4, 5, or more drugs concurrently (model capacity up to **15 active medications** per patient graph node cluster).
- **Pairwise Interaction Matrix Breakdown**: Automatically computes and evaluates all $\frac{N(N-1)}{2}$ drug pairs in a polypharmacy regimen, sorting them by risk probability and highlighting critical/major risk channels.
- **Select All / Batch Regimen Testing**: Quickly toggle active regimen subsets or evaluate the entire patient medication administration record (MAR) in a single click.

### 2. 🔬 RAG Mechanistic Explanation & Cross-Database Evidence Audit
- **Mechanistic Rationale**: Retrieves pharmacological context explaining metabolic pathway overlaps (e.g., CYP2C19 inhibition, CYP3A4 induction, renal competition).
- **Property Breakdown Table**: Structured breakdown mapping Pharmacokinetic (PK) shifts, Pharmacodynamic (PD) adverse event potentials, Mechanism pathways, and Receptor Targets.
- **Multi-Source Knowledge Hierarchy**: Cross-audits clinical guidelines and observational data across **DrugBank**, **CPIC Guidelines**, **TWOSIDES Observatory**, and **OFFSIDES Profiles**.

### 3. 🎛️ Segmented Control View Switching
- Dual dedicated dashboard views for streamlined clinical workflow:
  - **Segment 1: Medications & DDI Evaluation** — Focuses on active regimen MAR tables, multi-drug pairwise matrix, and DDI prediction risk panels.
  - **Segment 2: RAG Implementation & Evidence Audit** — Dedicated view for mechanistic summaries, property tables, and cross-database evidence trees.
  - **All Sections View** — Unified continuous view of all portal components.

### 4. 🌙 Light & Dark Theme Support
- Built-in theme toggle (`Light` / `Dark` mode) with automatic preference persistence via `localStorage`.
- High-contrast clinical EHR design with glassmorphism touches, curated HSL color palettes, and responsive layouts.

---

## 🚀 Getting Started

### Prerequisites
- **Node.js**: v18+ 
- **npm**: v9+
- **Python**: 3.9+ (for GNN models & graph scripts)

### Installation & Running Locally

1. **Clone Repository**
   ```bash
   git clone https://github.com/aditimaktedar/T_GNN_major_project.git
   cd T_GNN_major_project
   ```

2. **Frontend Development Server**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open `http://localhost:5173` (or the Vite assigned port) in your browser.

3. **Build for Production**
   ```bash
   npm run build
   ```

---

## 🏗️ Project Architecture

```
T_GNN_major_project/
├── data/                      # Raw, processed, and demo graph datasets
├── frontend/                  # React 19 + Vite EHR Portal
│   ├── src/
│   │   ├── components/
│   │   │   ├── PatientCard.jsx        # Patient demographics & navigation sidebar
│   │   │   ├── DrugPairCard.jsx       # Multi-drug regimen & pairwise DDI matrix
│   │   │   ├── PredictionPanel.jsx    # Polypharmacy interaction risk gauges
│   │   │   ├── RAGExplanation.jsx     # RAG mechanism & property table
│   │   │   └── EvidenceTree.jsx       # Cross-database evidence audit tree
│   │   ├── data/
│   │   │   └── mockPayload.js         # Polypharmacy evaluation engine & payload data
│   │   ├── App.jsx                    # Core app layout & segmented control router
│   │   └── index.css                  # Light & Dark CSS design system
│   └── package.json
├── notebooks/                 # Temporal graph demo & validation notebooks
├── src/
│   ├── graph/                 # Graph builder utilities & temporal edge extractors
│   ├── models/                # T-GNN model architecture & training scripts
│   └── xai/                   # Explainable AI (XAI) rationale generation
├── SCHEMA.md                  # Data schemas for graph nodes, edges, & features
└── README.md
```

---

## 🧪 Polypharmacy Model Specifications

| Parameter | Specification |
|---|---|
| **Graph Architecture** | Temporal Graph Neural Network (T-GNN) |
| **Max Regimen Capacity** | 15 Concurrent Active Medications |
| **Minimum Evaluation Limit** | 2 Medications |
| **Pairwise Calculation** | $O(N^2)$ Pairwise Combinatorial Subgraph Analysis |
| **Evidence Sources** | DrugBank, CPIC, TWOSIDES, OFFSIDES |

---

## 📄 License

This project is licensed under the MIT License - see the repository details for more information.
