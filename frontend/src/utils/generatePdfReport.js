import jsPDF from 'jspdf';

export function generatePdfReport(patient, selectedPair, activeEvaluatedData) {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 14;
  const contentWidth = pageWidth - margin * 2;
  let yPos = 14;

  // Primary palette
  const navy = [15, 23, 42];       // #0f172a
  const blue = [37, 99, 235];      // #2563eb
  const lightBlue = [239, 246, 255]; // #eff6ff
  const red = [220, 38, 38];       // #dc2626
  const grayText = [71, 85, 105];  // #475569
  const borderGray = [226, 232, 240]; // #e2e8f0

  // 1. Header Banner
  doc.setFillColor(...navy);
  doc.rect(margin, yPos, contentWidth, 22, 'F');

  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(14);
  doc.text('CLINICAL PATIENT & DDI EVALUATION REPORT', margin + 6, yPos + 9);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(191, 219, 254);
  doc.text('T-GNN Clinical Engine v2.4 • Confidential Medical Record', margin + 6, yPos + 16);

  // Top Right Timestamp
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  doc.setFontSize(8);
  doc.text(`Generated: ${dateStr} ${timeStr}`, pageWidth - margin - 6, yPos + 12, { align: 'right' });

  yPos += 28;

  // 2. Patient Profile Box
  doc.setFillColor(...lightBlue);
  doc.setDrawColor(...blue);
  doc.setLineWidth(0.4);
  doc.roundedRect(margin, yPos, contentWidth, 32, 2, 2, 'FD');

  doc.setTextColor(...navy);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(13);
  doc.text(patient.name || 'Patient Name', margin + 6, yPos + 8);

  // Demographic pills
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(...blue);
  doc.text(`AGE: ${patient.age} yrs`, margin + 6, yPos + 15);
  doc.setTextColor(...grayText);
  doc.setFont('helvetica', 'normal');
  doc.text(`|  GENDER: ${patient.gender || 'Female'}  |  DOB: ${patient.dob || '1954-07-22'}  |  MRN: ${patient.mrn || 'MRN-9842-7019'}`, margin + 30, yPos + 15);

  // Detail row 2
  doc.setFontSize(8.5);
  doc.setTextColor(...grayText);
  doc.text(`Attending Physician: ${patient.physician || 'Dr. Jordan Hughes'}`, margin + 6, yPos + 22);
  doc.text(`Location: ${patient.room || 'Bed 304-B (Step-down Unit)'}`, margin + 6, yPos + 27);
  doc.text(`Contact: ${patient.phone || '(406) 555-0120'}`, margin + 110, yPos + 22);
  doc.text(`Language: ${patient.language || 'English'}`, margin + 110, yPos + 27);

  yPos += 38;

  // 3. Evaluated DDI Interaction Summary Box
  const drug1 = selectedPair[0] || { name: 'Pantoprazole', id: 'DB00213' };
  const drug2 = selectedPair[1] || { name: 'Modafinil', id: 'DB00745' };
  const prediction = activeEvaluatedData?.prediction || {};

  const isCritical = prediction.severity === 'MAJOR' || prediction.severity === 'CRITICAL';
  const statusColor = isCritical ? red : blue;

  doc.setFillColor(isCritical ? 254 : 240, isCritical ? 242 : 249, isCritical ? 242 : 255);
  doc.setDrawColor(...statusColor);
  doc.setLineWidth(0.5);
  doc.roundedRect(margin, yPos, contentWidth, 34, 2, 2, 'FD');

  doc.setTextColor(...statusColor);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.text('EVALUATED DRUG PAIR INTERACTION RISK', margin + 6, yPos + 7);

  doc.setTextColor(...navy);
  doc.setFontSize(11);
  doc.text(`${drug1.name} (${drug1.id})   ↔   ${drug2.name} (${drug2.id})`, margin + 6, yPos + 15);

  // Metrics columns
  // Col 1: Status
  doc.setFontSize(8);
  doc.setTextColor(...grayText);
  doc.text('INTERACTION STATUS', margin + 6, yPos + 22);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.setTextColor(...statusColor);
  doc.text(prediction.interaction === 'YES' ? 'YES — INTERACTION DETECTED' : 'NO INTERACTION DETECTED', margin + 6, yPos + 28);

  // Col 2: Probability
  const probVal = prediction.probability ? Math.round(prediction.probability * 100) : 91;
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(...grayText);
  doc.text('MODEL PROBABILITY', margin + 75, yPos + 22);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.setTextColor(...navy);
  doc.text(`${probVal}% Confidence`, margin + 75, yPos + 28);

  // Col 3: Severity
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(...grayText);
  doc.text('SEVERITY LEVEL', margin + 130, yPos + 22);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.setTextColor(...statusColor);
  doc.text(`${prediction.severity || 'MAJOR'} SEVERITY`, margin + 130, yPos + 28);

  yPos += 40;

  // 4. RAG Clinical Explanation
  doc.setFillColor(248, 250, 252);
  doc.setDrawColor(...borderGray);
  doc.setLineWidth(0.3);
  doc.roundedRect(margin, yPos, contentWidth, 38, 2, 2, 'FD');

  doc.setTextColor(...navy);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.text('RAG CLINICAL MECHANISM EXPLANATION', margin + 6, yPos + 7);

  const ragData = activeEvaluatedData?.rag_explanation || {};
  const summaryText = ragData.summary || "Pantoprazole and Modafinil exhibit significant metabolic pathway overlap in hepatic biotransformation.";
  
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.setTextColor(...grayText);

  // Wrap summary text
  const splitSummary = doc.splitTextToSize(summaryText, contentWidth - 12);
  doc.text(splitSummary.slice(0, 3), margin + 6, yPos + 13);

  // Shared mechanisms pills
  const mechanisms = ragData.shared_mechanisms || ["Hepatic CYP2C19 Inhibition", "CYP3A4 Enzyme Induction"];
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(...navy);
  doc.text(`Mechanisms: ${mechanisms.join(' • ')}`, margin + 6, yPos + 32);

  yPos += 44;

  // 5. Active MAR Regimen Table
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(...navy);
  doc.text('ACTIVE MEDICATION ADMINISTRATION RECORD (MAR)', margin, yPos);
  yPos += 4;

  // Table header
  doc.setFillColor(241, 245, 249);
  doc.setDrawColor(...borderGray);
  doc.rect(margin, yPos, contentWidth, 7, 'FD');

  doc.setFontSize(8);
  doc.setTextColor(...grayText);
  doc.text('DRUGBANK ID', margin + 4, yPos + 5);
  doc.text('MEDICATION NAME', margin + 35, yPos + 5);
  doc.text('DOSAGE', margin + 95, yPos + 5);
  doc.text('FREQUENCY', margin + 125, yPos + 5);
  doc.text('ROUTE', margin + 160, yPos + 5);

  yPos += 7;

  // Table rows
  const meds = patient.medications || [];
  meds.forEach((med, idx) => {
    if (yPos > pageHeight - 25) {
      doc.addPage();
      yPos = 14;
    }

    const isSelected = selectedPair.some(m => m.id === med.id);
    if (isSelected) {
      doc.setFillColor(239, 246, 255);
      doc.rect(margin, yPos, contentWidth, 7, 'F');
    }

    doc.setDrawColor(...borderGray);
    doc.line(margin, yPos + 7, margin + contentWidth, yPos + 7);

    doc.setFont('helvetica', isSelected ? 'bold' : 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(isSelected ? blue[0] : navy[0], isSelected ? blue[1] : navy[1], isSelected ? blue[2] : navy[2]);
    doc.text(med.id || `DB0000${idx}`, margin + 4, yPos + 5);

    doc.setTextColor(...navy);
    doc.text(med.name, margin + 35, yPos + 5);
    if (isSelected) {
      doc.setFontSize(7);
      doc.setTextColor(...blue);
      doc.text('[Evaluated Pair]', margin + 70, yPos + 5);
    }

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(...grayText);
    doc.text(med.dosage || '40 mg', margin + 95, yPos + 5);
    doc.text(med.frequency || 'QD (Daily)', margin + 125, yPos + 5);
    doc.text(med.route || 'Oral', margin + 160, yPos + 5);

    yPos += 7;
  });

  yPos += 8;

  // 6. Evidence Audit Section
  if (yPos > pageHeight - 30) {
    doc.addPage();
    yPos = 14;
  }

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.setTextColor(...navy);
  doc.text('MULTI-DATABASE EVIDENCE AUDIT SUMMARY', margin, yPos);
  yPos += 4;

  const ev = activeEvaluatedData?.evidence || {};
  const dbCount = (ev.drugbank?.drug_1_count || 0) + (ev.drugbank?.drug_2_count || 0);
  const cpicCount = (ev.cpic?.drug_1_count || 0) + (ev.cpic?.drug_2_count || 0);

  doc.setFillColor(248, 250, 252);
  doc.roundedRect(margin, yPos, contentWidth, 16, 2, 2, 'FD');

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(...grayText);
  doc.text(`• DrugBank Database: ${dbCount} interactions logged`, margin + 6, yPos + 6);
  doc.text(`• CPIC Pharmacogenomics: ${cpicCount} clinical guidelines found`, margin + 6, yPos + 11);
  doc.text(`• TwoSides Pharmacovigilance: ${ev.twosides?.observed ? 'Observed in FDA FAERS data' : 'No prior FAERS reports'}`, margin + 95, yPos + 6);
  doc.text(`• OffSides Signal Detection: ${ev.offsides?.drug_1_available ? 'Active signals recorded' : 'Standard profile'}`, margin + 95, yPos + 11);

  yPos += 22;

  // Page Footer
  doc.setFont('helvetica', 'italic');
  doc.setFontSize(7.5);
  doc.setTextColor(148, 163, 184);
  doc.text(`Report generated for ${patient.name} (MRN: ${patient.mrn}) • T-GNN Clinical Portal • Page 1 of 1`, pageWidth / 2, pageHeight - 8, { align: 'center' });

  // Save the PDF file
  const filename = `Patient_Report_${patient.name ? patient.name.replace(/[^a-zA-Z0-9]/g, '_') : 'Patient'}_${patient.mrn || 'MRN'}.pdf`;
  doc.save(filename);
}
