import jsPDF from 'jspdf';
import { normalizePatientForReport, prepareEvaluationForReport } from './reportData';

const MARGIN = 18;
const PAGE_BOTTOM = 272;
const LINE = 4.8;

const COLORS = {
  text: [44, 44, 44],
  muted: [107, 107, 107],
  sage: [95, 125, 115],
  border: [210, 208, 203],
  white: [255, 255, 255],
};

function fmt(value, emptyLabel = 'Not recorded') {
  if (value === null || value === undefined || value === '') return emptyLabel;
  return String(value);
}

function drugId(med) {
  return med?.id || med?.drugbank_id || '—';
}

function formatReportTimestamp(date) {
  return date.toLocaleString('en-GB', {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

function ensureSpace(doc, y, needed, margin, onNewPage) {
  if (y + needed <= PAGE_BOTTOM) return y;
  doc.addPage();
  if (onNewPage) onNewPage();
  return margin + 8;
}

function drawHorizontalRule(doc, y, pageWidth, margin) {
  doc.setDrawColor(...COLORS.border);
  doc.setLineWidth(0.2);
  doc.line(margin, y, pageWidth - margin, y);
}

function drawSectionTitle(doc, title, y, margin) {
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(...COLORS.sage);
  doc.text(title, margin, y);
  doc.setTextColor(...COLORS.text);
  return y + 6;
}

function drawBodyText(doc, text, y, margin, contentWidth) {
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.text);
  const lines = doc.splitTextToSize(text, contentWidth);
  lines.forEach((line) => {
    y = ensureSpace(doc, y, LINE, margin);
    doc.text(line, margin, y);
    y += LINE;
  });
  return y;
}

function drawKeyValueRow(doc, label, value, y, margin, contentWidth) {
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.muted);
  doc.text(label, margin, y);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...COLORS.text);
  const valLines = doc.splitTextToSize(fmt(value), contentWidth - 42);
  doc.text(valLines[0], margin + 42, y);
  y += LINE;
  for (let i = 1; i < valLines.length; i += 1) {
    y = ensureSpace(doc, y, LINE, margin);
    doc.text(valLines[i], margin + 42, y);
    y += LINE;
  }
  return y;
}

function drawTableHeader(doc, columns, y, margin) {
  doc.setFillColor(248, 247, 245);
  doc.setDrawColor(...COLORS.border);
  const rowH = 7;
  const tableW = columns.reduce((s, c) => s + c.width, 0);
  doc.rect(margin, y - 4.5, tableW, rowH, 'FD');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(...COLORS.muted);
  let x = margin + 2;
  columns.forEach((col) => {
    doc.text(col.title, x, y);
    x += col.width;
  });
  return y + 4;
}

function drawTableRow(doc, cells, columns, y, margin, rowHeight = 7) {
  const tableW = columns.reduce((s, c) => s + c.width, 0);
  doc.setDrawColor(...COLORS.border);
  doc.setLineWidth(0.15);
  doc.line(margin, y + rowHeight - 4, margin + tableW, y + rowHeight - 4);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.setTextColor(...COLORS.text);
  let x = margin + 2;
  columns.forEach((col, idx) => {
    const raw = cells[idx] ?? '—';
    const clipped = doc.splitTextToSize(String(raw), col.width - 4);
    doc.text(clipped[0], x, y);
    x += col.width;
  });
  return y + rowHeight;
}

function addPageFooters(doc, patient, generatedAt) {
  const pageCount = doc.internal.getNumberOfPages();
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  for (let i = 1; i <= pageCount; i += 1) {
    doc.setPage(i);
    drawHorizontalRule(doc, pageHeight - 14, pageWidth, MARGIN);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(...COLORS.muted);
    const left = fmt(patient?.mrn, 'Patient record');
    doc.text(`DDI Research Portal · ${left}`, MARGIN, pageHeight - 9);
    doc.text(`Generated ${formatReportTimestamp(generatedAt)}`, pageWidth / 2, pageHeight - 9, {
      align: 'center',
    });
    doc.text(`Page ${i} of ${pageCount}`, pageWidth - MARGIN, pageHeight - 9, { align: 'right' });
  }
}

/**
 * @param {{ patient: object, evaluation: object|null, currentSelection: object[], engineHealth: object|null }} input
 */
export async function generatePdfReport({ patient, evaluation, currentSelection, engineHealth }) {
  if (!patient || !patient.name) {
    throw new Error('Patient record is unavailable. Save the profile or connect to the API before downloading.');
  }

  const patientRecord = normalizePatientForReport(patient);
  const evaluationPrepared = prepareEvaluationForReport(evaluation, patientRecord);

  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const pageWidth = doc.internal.pageSize.getWidth();
  const contentWidth = pageWidth - MARGIN * 2;
  const generatedAt = new Date();
  let y = MARGIN;

  const onNewPage = () => {
    y = MARGIN + 4;
  };

  // Report header
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16);
  doc.setTextColor(...COLORS.text);
  doc.text('Drug–Drug Interaction Evaluation Report', MARGIN, y);
  y += 8;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.muted);
  doc.text(`Patient: ${patientRecord.name}`, MARGIN, y);
  y += LINE;
  doc.text(`Identifier: ${fmt(patientRecord.mrn)}`, MARGIN, y);
  y += LINE;
  doc.text(`Report generated: ${formatReportTimestamp(generatedAt)}`, MARGIN, y);
  y += LINE;

  const evalStatus = evaluationPrepared ? 'Evaluation completed' : 'No current evaluation included';
  doc.text(`Evaluation status: ${evalStatus}`, MARGIN, y);
  y += LINE + 2;

  const disclaimer =
    'This document summarizes research-model predictions and retrieved reference text. It is not a clinical diagnosis, prescribing guide, or substitute for review by a licensed clinician.';
  y = drawBodyText(doc, disclaimer, y, MARGIN, contentWidth);
  y += 4;
  drawHorizontalRule(doc, y, pageWidth, MARGIN);
  y += 8;

  // 1. Patient information
  y = drawSectionTitle(doc, '1. Patient information', y, MARGIN);
  y = drawKeyValueRow(doc, 'Full name:', patientRecord.name, y, MARGIN, contentWidth);
  y = drawKeyValueRow(doc, 'Date of birth:', patientRecord.dob, y, MARGIN, contentWidth);
  y = drawKeyValueRow(doc, 'Age (years):', patientRecord.age, y, MARGIN, contentWidth);
  if (patientRecord.dob && patientRecord.age != null) {
    y = drawBodyText(
      doc,
      'Age is calculated from date of birth using the same rule as the clinical portal.',
      y,
      MARGIN,
      contentWidth,
    );
  }
  y = drawKeyValueRow(doc, 'Gender:', patientRecord.gender, y, MARGIN, contentWidth);
  y = drawKeyValueRow(doc, 'Preferred language:', patientRecord.language, y, MARGIN, contentWidth);
  y = drawKeyValueRow(doc, 'Weight:', patientRecord.weight, y, MARGIN, contentWidth);
  y = drawKeyValueRow(doc, 'Care location:', patientRecord.room, y, MARGIN, contentWidth);
  y = drawKeyValueRow(doc, 'Attending physician:', patientRecord.physician, y, MARGIN, contentWidth);
  y += 4;

  // 2. Medication regimen
  y = ensureSpace(doc, y, 20, MARGIN, onNewPage);
  y = drawSectionTitle(doc, '2. Medication regimen', y, MARGIN);

  const evalDrugs = evaluationPrepared?.selectedDrugs || [];
  const selection = currentSelection || [];
  const selectionSig = (list) =>
    list
      .map((d) => d.id || d.drugbank_id || d.name || '')
      .filter(Boolean)
      .sort()
      .join('|');
  const selectionMatchesEval =
    evaluationPrepared && selectionSig(selection) === selectionSig(evalDrugs);

  if (evaluationPrepared && evalDrugs.length > 0) {
    y = drawBodyText(
      doc,
      selectionMatchesEval
        ? 'Medications included in the evaluation reported below.'
        : 'Evaluation was run on a different selection than the current portal selection; regimen table reflects the evaluated set.',
      y,
      MARGIN,
      contentWidth,
    );
    y += 2;
  } else {
    y = drawBodyText(
      doc,
      'No evaluation is attached to this report. The table lists the patient’s active medication record from the clinical database.',
      y,
      MARGIN,
      contentWidth,
    );
    y += 2;
  }

  const regimenSource = evaluationPrepared?.selectedDrugs?.length
    ? evaluationPrepared.selectedDrugs
    : patientRecord.medications || [];
  const fullMar = patientRecord.medications || [];
  const evalIds = new Set(regimenSource.map((m) => drugId(m)));

  const medCols = [
    { title: 'DrugBank ID', width: 26 },
    { title: 'Medication', width: 48 },
    { title: 'Dosage', width: 26 },
    { title: 'Frequency', width: 30 },
    { title: 'Route', width: 20 },
    { title: 'In evaluation', width: 24 },
  ];

  const drawMedHeader = () => {
    y = ensureSpace(doc, y, 12, MARGIN, onNewPage);
    y = drawTableHeader(doc, medCols, y, MARGIN);
  };

  drawMedHeader();
  if (regimenSource.length === 0) {
    y = drawBodyText(doc, 'No medications on record.', y, MARGIN, contentWidth);
  } else {
    regimenSource.forEach((med) => {
      y = ensureSpace(doc, y, 8, MARGIN, onNewPage);
      if (y <= MARGIN + 10) drawMedHeader();
      y = drawTableRow(
        doc,
        [
          drugId(med),
          med.name || '—',
          fmt(med.dosage, '—'),
          fmt(med.frequency, '—'),
          fmt(med.route, '—'),
          evalIds.has(drugId(med)) ? 'Yes' : 'No',
        ],
        medCols,
        y,
        MARGIN,
      );
    });
  }

  if (fullMar.length > regimenSource.length) {
    y += 3;
    y = drawBodyText(
      doc,
      `Complete active regimen (${fullMar.length} medications) is maintained in the portal; ${regimenSource.length} medication(s) are listed above for this report context.`,
      y,
      MARGIN,
      contentWidth,
    );
  }
  y += 6;

  // 3. Evaluation summary
  y = ensureSpace(doc, y, 24, MARGIN, onNewPage);
  y = drawSectionTitle(doc, '3. Evaluation summary', y, MARGIN);

  if (!evaluationPrepared) {
    y = drawBodyText(
      doc,
      'No interaction evaluation is included. Run an evaluation in the portal and download again to attach pairwise results.',
      y,
      MARGIN,
      contentWidth,
    );
  } else {
    const pred = evaluationPrepared.prediction || {};
    const meta = evaluationPrepared.evaluation_meta || {};
    const engineLabel =
      pred.source ||
      meta.engine ||
      (engineHealth?.engine?.model ? String(engineHealth.engine.model) : 'TemporalDDI-GNN');
    const selectedCount = evalDrugs.length || selection.length;
    y = drawKeyValueRow(doc, 'Model / source:', engineLabel, y, MARGIN, contentWidth);
    y = drawKeyValueRow(doc, 'Medications evaluated:', selectedCount, y, MARGIN, contentWidth);
    y = drawKeyValueRow(
      doc,
      'Pairwise channels:',
      evaluationPrepared.pairCount ?? (evaluationPrepared.pairs || []).length,
      y,
      MARGIN,
      contentWidth,
    );
    if (evaluationPrepared.expectedPairCount != null) {
      y = drawKeyValueRow(
        doc,
        'Expected pairwise channels:',
        evaluationPrepared.expectedPairCount,
        y,
        MARGIN,
        contentWidth,
      );
    }
    if (evaluationPrepared.countMismatch) {
      y = drawBodyText(
        doc,
        'Note: the number of pairwise rows in this report does not match the expected count for the evaluated regimen. Verify the evaluation response before clinical use.',
        y,
        MARGIN,
        contentWidth,
      );
    }
    y = drawKeyValueRow(
      doc,
      'Aggregate interaction signal:',
      pred.interaction != null ? String(pred.interaction) : 'Not recorded',
      y,
      MARGIN,
      contentWidth,
    );
    if (pred.probability != null) {
      y = drawKeyValueRow(
        doc,
        'Highest pairwise model score:',
        `${Math.round(Number(pred.probability) * 100)}%`,
        y,
        MARGIN,
        contentWidth,
      );
    }
    if (pred.severity) {
      y = drawKeyValueRow(doc, 'Highest severity label:', pred.severity, y, MARGIN, contentWidth);
    }
    if (
      evaluationPrepared.majorCount != null ||
      evaluationPrepared.modCount != null ||
      evaluationPrepared.lowCount != null
    ) {
      y = drawKeyValueRow(
        doc,
        'Severity distribution:',
        `Major: ${evaluationPrepared.majorCount ?? 0}, Moderate: ${evaluationPrepared.modCount ?? 0}, Low: ${evaluationPrepared.lowCount ?? 0}`,
        y,
        MARGIN,
        contentWidth,
      );
    }
    if (pred.alertCode) {
      y = drawKeyValueRow(doc, 'Alert reference:', pred.alertCode, y, MARGIN, contentWidth);
    }
    y += 2;
    y = drawBodyText(
      doc,
      'Model scores indicate statistical association from the research pipeline; they are not calibrated probabilities of clinical harm.',
      y,
      MARGIN,
      contentWidth,
    );
  }
  y += 6;

  // 4. Pairwise results
  y = ensureSpace(doc, y, 20, MARGIN, onNewPage);
  y = drawSectionTitle(doc, '4. Pairwise interaction results', y, MARGIN);

  const pairs = evaluationPrepared?.pairs || [];
  const unresolvedPairs = pairs.filter(
    (p) => p.drug1?.nameUnresolved || p.drug2?.nameUnresolved,
  ).length;

  if (!evaluationPrepared || pairs.length === 0) {
    y = drawBodyText(doc, 'No pairwise results available.', y, MARGIN, contentWidth);
  } else {
    const pairCols = [
      { title: 'Drug A name', width: 26 },
      { title: 'Drug A ID', width: 18 },
      { title: 'Drug B name', width: 26 },
      { title: 'Drug B ID', width: 18 },
      { title: 'Signal', width: 16 },
      { title: 'Score', width: 14 },
      { title: 'Severity', width: 16 },
      { title: 'Mechanism', width: 40 },
    ];

    const drawPairHeader = () => {
      y = ensureSpace(doc, y, 12, MARGIN, onNewPage);
      y = drawTableHeader(doc, pairCols, y, MARGIN);
    };

    drawPairHeader();
    if (unresolvedPairs > 0) {
      y = drawBodyText(
        doc,
        `${unresolvedPairs} pair row(s) could not be mapped to a clinical medication name from the regimen; identifiers are shown separately.`,
        y,
        MARGIN,
        contentWidth,
      );
      y += 2;
    }

    pairs.forEach((pair) => {
      const mech = pair.mechanism || pair.summary || '';
      const mechShort = mech.length > 100 ? `${mech.slice(0, 97)}…` : mech;
      const score =
        pair.probability != null ? `${Math.round(Number(pair.probability) * 100)}%` : '—';
      const nameA = pair.drug1?.name || '—';
      const nameB = pair.drug2?.name || '—';
      const idA = pair.drug1?.id || '—';
      const idB = pair.drug2?.id || '—';
      y = ensureSpace(doc, y, 10, MARGIN, onNewPage);
      if (y <= MARGIN + 12) drawPairHeader();
      y = drawTableRow(
        doc,
        [
          nameA,
          idA,
          nameB,
          idB,
          pair.interaction || '—',
          score,
          pair.severity || '—',
          mechShort || '—',
        ],
        pairCols,
        y,
        MARGIN,
        8,
      );
      if (pair.summary && pair.summary !== mech) {
        y = ensureSpace(doc, y, LINE * 2, MARGIN, onNewPage);
        doc.setFontSize(8);
        doc.setTextColor(...COLORS.muted);
        const detail = doc.splitTextToSize(pair.summary, contentWidth - 4);
        detail.slice(0, 4).forEach((line) => {
          y = ensureSpace(doc, y, LINE, MARGIN, onNewPage);
          doc.text(line, MARGIN + 2, y);
          y += LINE - 0.5;
        });
        doc.setTextColor(...COLORS.text);
      }
    });
  }
  y += 6;

  // 5. Supporting evidence
  y = ensureSpace(doc, y, 20, MARGIN, onNewPage);
  y = drawSectionTitle(doc, '5. Supporting evidence (retrieved context)', y, MARGIN);

  if (!evaluationPrepared) {
    y = drawBodyText(doc, 'Not available — no evaluation included.', y, MARGIN, contentWidth);
  } else {
    const rag = evaluationPrepared.rag_explanation || {};
    if (rag.summary) {
      y = drawBodyText(doc, 'Regimen-level retrieved summary (RAG):', y, MARGIN, contentWidth);
      y = drawBodyText(doc, rag.summary, y, MARGIN, contentWidth);
      y += 2;
    }
    const mechList = rag.shared_mechanisms || [];
    if (mechList.length > 0) {
      y = drawBodyText(doc, `Shared mechanisms cited: ${mechList.join('; ')}`, y, MARGIN, contentWidth);
    }
    const ev = evaluationPrepared.evidence || {};
    if (Object.keys(ev).length > 0) {
      y += 2;
      y = drawBodyText(doc, 'Source audit aggregates for this evaluation:', y, MARGIN, contentWidth);
      if (ev.drugbank?.total != null) {
        y = drawKeyValueRow(doc, 'DrugBank entries (total):', ev.drugbank.total, y, MARGIN, contentWidth);
      }
      if (ev.cpic?.total != null) {
        y = drawKeyValueRow(doc, 'CPIC guideline hits (total):', ev.cpic.total, y, MARGIN, contentWidth);
      }
      if (ev.twosides && 'observed' in ev.twosides) {
        y = drawKeyValueRow(
          doc,
          'TwoSides FAERS signal:',
          ev.twosides.observed ? 'Observed in aggregated data' : 'Not observed in aggregated data',
          y,
          MARGIN,
          contentWidth,
        );
      }
    }
    if (!rag.summary && !mechList.length && !Object.keys(ev).length) {
      y = drawBodyText(doc, 'No supporting evidence text was returned for this evaluation.', y, MARGIN, contentWidth);
    }
    y += 2;
    y = drawBodyText(
      doc,
      'Retrieved passages support interpretation; they are not independent clinical verification of the model output.',
      y,
      MARGIN,
      contentWidth,
    );
  }
  y += 6;

  // 6. Methodology and limitations
  y = ensureSpace(doc, y, 30, MARGIN, onNewPage);
  y = drawSectionTitle(doc, '6. Methodology and limitations', y, MARGIN);
  const methodology = [
    'This portal combines a Temporal Graph Neural Network (TemporalDDI-GNN) regimen evaluator with retrieval-augmented generation (RAG) over curated drug and interaction knowledge sources.',
    'Pairwise predictions are produced for each drug pair in the selected regimen; aggregate summary statistics reflect the highest-scoring pair and severity distribution across pairs.',
    'Outputs are intended for clinical research and decision support prototyping only. Predictions require independent review against primary literature, formularies, and patient-specific factors.',
    'The system does not assert regulatory approval, prospective clinical validation, or proven benefit for any individual patient.',
  ];
  methodology.forEach((para) => {
    y = drawBodyText(doc, para, y, MARGIN, contentWidth);
    y += 2;
  });
  y += 4;

  // 7. Report provenance
  y = ensureSpace(doc, y, 20, MARGIN, onNewPage);
  y = drawSectionTitle(doc, '7. Report provenance', y, MARGIN);
  y = drawKeyValueRow(doc, 'Report generated (local):', formatReportTimestamp(generatedAt), y, MARGIN, contentWidth);
  y = drawKeyValueRow(doc, 'Patient identifier:', patientRecord.mrn, y, MARGIN, contentWidth);
  if (evaluationPrepared?.prediction?.alertCode) {
    y = drawKeyValueRow(doc, 'Evaluation alert reference:', evaluationPrepared.prediction.alertCode, y, MARGIN, contentWidth);
  }
  if (engineHealth?.engine?.status) {
    y = drawKeyValueRow(doc, 'Engine status at generation:', engineHealth.engine.status, y, MARGIN, contentWidth);
  }

  addPageFooters(doc, patientRecord, generatedAt);

  const safeName = patientRecord.name.replace(/[^a-zA-Z0-9_-]+/g, '_').slice(0, 48);
  const safeMrn = (patientRecord.mrn || 'record').replace(/[^a-zA-Z0-9_-]+/g, '_');
  const filename = `DDI_Report_${safeName}_${safeMrn}.pdf`;
  doc.save(filename);
}
