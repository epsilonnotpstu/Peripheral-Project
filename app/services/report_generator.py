"""
PDF report generator using ReportLab.

Generates 6-page clinical reports from session data stored in SQLite.
Matplotlib figures are embedded as PNG bytes (no temp files needed).

Report structure:
  Page 1: Cover — patient info + session timestamps
  Page 2: Summary statistics + beat distribution pie chart
  Page 3: ECG strip (last 10 seconds, filtered, with R-peak markers)
  Page 4: AI prediction timeline (beat index vs class, colored by confidence)
  Page 5: Per-class statistics table
  Page 6: Clinical disclaimer
"""

import io
import os
import logging
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Line
from reportlab.graphics.charts.piecharts import Pie

log = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
SHORT_NAMES = ["N", "S", "V", "F", "Q"]
CLASS_COLORS_HEX = ["#4CAF50", "#2196F3", "#F44336", "#FF9800", "#9E9E9E"]


class ReportGenerator:
    """Generates professional PDF clinical reports for completed ECG sessions."""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def generate(self, session_id: int) -> str:
        """
        Generate a PDF report for the given session.

        Args:
            session_id: ID of the RecordingSession to report on.

        Returns:
            Absolute path to the generated PDF file.

        Raises:
            ValueError: If session or patient not found.
        """
        from app.extensions import db
        from app.models.session import RecordingSession
        from app.models.prediction import Prediction
        from app.models.ecg_record import ECGRecord

        # ── Fetch data ────────────────────────────────────────────────────────
        session = RecordingSession.query.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        patient = session.patient
        predictions = session.predictions.order_by(Prediction.timestamp).all()
        ecg_records = session.ecg_records.order_by(ECGRecord.timestamp).all()

        # ── Output path ───────────────────────────────────────────────────────
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"ECG_Report_Patient{patient.id}_Session{session_id}_{ts}.pdf"
        output_path = os.path.join(self.reports_dir, fname)

        # ── Build PDF ─────────────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        styles = getSampleStyleSheet()
        story = []

        story += self._page_cover(patient, session, styles)
        story.append(PageBreak())
        story += self._page_summary(predictions, styles)
        story.append(PageBreak())
        story += self._page_ecg_strip(ecg_records, predictions, styles)
        story.append(PageBreak())
        story += self._page_prediction_timeline(predictions, styles)
        story.append(PageBreak())
        story += self._page_class_table(predictions, styles)
        story.append(PageBreak())
        story += self._page_disclaimer(styles)

        doc.build(story)
        log.info(f"PDF report generated: {output_path}")
        return output_path

    # ── Page builders ──────────────────────────────────────────────────────────

    def _page_cover(self, patient, session, styles) -> list:
        heading = ParagraphStyle("Heading", fontSize=20, fontName="Helvetica-Bold",
                                  alignment=TA_CENTER, spaceAfter=6)
        subheading = ParagraphStyle("Sub", fontSize=14, fontName="Helvetica",
                                     alignment=TA_CENTER, spaceAfter=12, textColor=colors.gray)
        field = ParagraphStyle("Field", fontSize=11, fontName="Helvetica", spaceAfter=4)

        story = [
            Spacer(1, 1.5*cm),
            Paragraph("ECG Monitoring Report", heading),
            Paragraph("AI-Assisted Arrhythmia Analysis", subheading),
            HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2196F3")),
            Spacer(1, 1*cm),
        ]

        # Patient info table
        data = [
            ["Patient Name", patient.name],
            ["Patient ID", patient.medical_id or "N/A"],
            ["Age", str(patient.age) if patient.age else "N/A"],
            ["Gender", patient.gender or "N/A"],
            ["Session ID", str(session.id)],
            ["Recording Start", session.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if session.started_at else "N/A"],
            ["Recording End", session.ended_at.strftime("%Y-%m-%d %H:%M:%S UTC") if session.ended_at else "Ongoing"],
            ["Duration", f"{session.duration_s:.0f} s ({session.duration_s/60:.1f} min)" if session.duration_s else "N/A"],
            ["Notes", session.notes or "None"],
            ["Report Generated", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")],
        ]

        table = Table(data, colWidths=[5*cm, 11*cm])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E3F2FD")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BDBDBD")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))

        story.append(table)
        return story

    def _page_summary(self, predictions, styles) -> list:
        story = [Paragraph("Session Summary", styles["Heading1"]), Spacer(1, 0.5*cm)]

        if not predictions:
            story.append(Paragraph("No predictions recorded for this session.", styles["Normal"]))
            return story

        # Count per class
        counts = {n: 0 for n in CLASS_NAMES}
        confidences = {n: [] for n in CLASS_NAMES}
        bpms = []
        motion_count = 0
        alert_count = 0

        for p in predictions:
            counts[p.class_name] = counts.get(p.class_name, 0) + 1
            confidences[p.class_name].append(p.confidence)
            if p.bpm:
                bpms.append(p.bpm)
            if p.motion_flag:
                motion_count += 1
            if p.alert_raised:
                alert_count += 1

        total = len(predictions)
        avg_bpm = np.mean(bpms) if bpms else 0.0
        min_bpm = np.min(bpms) if bpms else 0.0
        max_bpm = np.max(bpms) if bpms else 0.0

        # Summary stats table
        summary_data = [
            ["Total Beats Analyzed", str(total)],
            ["Average BPM", f"{avg_bpm:.1f}"],
            ["Min / Max BPM", f"{min_bpm:.1f} / {max_bpm:.1f}"],
            ["Motion Artifact Beats", f"{motion_count} ({motion_count/total*100:.1f}%)"],
            ["Clinical Alerts Raised", str(alert_count)],
        ]
        tbl = Table(summary_data, colWidths=[7*cm, 9*cm])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
            ("BOX", (0, 0), (-1, -1), 1, colors.lightgrey),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.8*cm))

        # Pie chart of beat distribution
        pie_img = self._make_pie_chart(counts, total)
        if pie_img:
            story.append(Paragraph("Beat Classification Distribution", styles["Heading2"]))
            story.append(pie_img)

        return story

    def _make_pie_chart(self, counts: dict, total: int):
        if total == 0:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        non_zero = [(n, c) for n, c in counts.items() if c > 0]
        labels = [f"{n}\n{c} ({c/total*100:.1f}%)" for n, c in non_zero]
        sizes = [c for _, c in non_zero]
        color_map = dict(zip(CLASS_NAMES, CLASS_COLORS_HEX))
        pie_colors = [color_map.get(n, "#757575") for n, _ in non_zero]

        wedges, texts = ax.pie(sizes, labels=labels, colors=pie_colors,
                                startangle=140, textprops={"fontsize": 9})
        ax.set_title("Beat Classification Distribution", fontsize=12, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return Image(buf, width=14*cm, height=9*cm)

    def _page_ecg_strip(self, ecg_records, predictions, styles) -> list:
        story = [Paragraph("ECG Strip (Last 10 Seconds)", styles["Heading1"]), Spacer(1, 0.3*cm)]

        # Reconstruct last 10 seconds from ECGRecord chunks
        all_samples = []
        for rec in ecg_records[-10:]:   # last 10 records = last 10 seconds
            all_samples.extend(rec.get_samples())

        if not all_samples:
            story.append(Paragraph("No ECG data available.", styles["Normal"]))
            return story

        fs = 125
        t = np.arange(len(all_samples)) / fs

        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.plot(t, all_samples, color="#00BCD4", linewidth=0.8, alpha=0.9)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Amplitude (normalized)", fontsize=10)
        ax.set_title("Filtered ECG Signal — Last 10 Seconds", fontsize=11)
        ax.set_xlim([t[0], t[-1]])
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        story.append(Image(buf, width=16*cm, height=5.5*cm))
        return story

    def _page_prediction_timeline(self, predictions, styles) -> list:
        story = [Paragraph("AI Prediction Timeline", styles["Heading1"]), Spacer(1, 0.3*cm)]

        if len(predictions) < 2:
            story.append(Paragraph("Insufficient predictions for timeline.", styles["Normal"]))
            return story

        beat_indices = [p.beat_index or i for i, p in enumerate(predictions)]
        class_ids = [p.class_id for p in predictions]
        confidences = [p.confidence for p in predictions]

        color_map = {0: "#4CAF50", 1: "#2196F3", 2: "#F44336", 3: "#FF9800", 4: "#9E9E9E"}
        point_colors = [color_map.get(c, "#9E9E9E") for c in class_ids]

        fig, ax = plt.subplots(figsize=(14, 4))
        scatter = ax.scatter(beat_indices, class_ids, c=confidences, cmap="RdYlGn",
                              s=20, alpha=0.7, vmin=0.5, vmax=1.0)
        plt.colorbar(scatter, ax=ax, label="Confidence", shrink=0.8)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Beat Index", fontsize=10)
        ax.set_title("AI Classification Per Beat (color = confidence)", fontsize=11)
        ax.grid(True, alpha=0.2, axis="x")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        story.append(Image(buf, width=16*cm, height=6.5*cm))
        return story

    def _page_class_table(self, predictions, styles) -> list:
        story = [Paragraph("Per-Class Statistics", styles["Heading1"]), Spacer(1, 0.4*cm)]

        counts = {n: 0 for n in CLASS_NAMES}
        conf_lists = {n: [] for n in CLASS_NAMES}
        alerts = {n: 0 for n in CLASS_NAMES}

        for p in predictions:
            counts[p.class_name] = counts.get(p.class_name, 0) + 1
            conf_lists[p.class_name].append(p.confidence)
            if p.alert_raised:
                alerts[p.class_name] = alerts.get(p.class_name, 0) + 1

        total = max(len(predictions), 1)

        header = ["Class", "Full Name", "Count", "%", "Avg Conf.", "Alerts"]
        rows = [header]
        for short, full in zip(SHORT_NAMES, CLASS_NAMES):
            n = counts[full]
            avg_c = np.mean(conf_lists[full]) if conf_lists[full] else 0.0
            rows.append([
                short, full, str(n), f"{n/total*100:.1f}%",
                f"{avg_c:.3f}", str(alerts[full]),
            ])

        tbl = Table(rows, colWidths=[1.2*cm, 4.5*cm, 2*cm, 2*cm, 2.5*cm, 2*cm])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2FF")]),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BDBDBD")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ]))
        story.append(tbl)
        return story

    def _page_disclaimer(self, styles) -> list:
        disclaimer_style = ParagraphStyle(
            "Disclaimer", fontSize=10, fontName="Helvetica",
            alignment=TA_JUSTIFY, leading=14, spaceAfter=10,
        )
        return [
            Spacer(1, 3*cm),
            HRFlowable(width="100%", thickness=1, color=colors.gray),
            Spacer(1, 0.5*cm),
            Paragraph("CLINICAL DISCLAIMER", styles["Heading2"]),
            Spacer(1, 0.3*cm),
            Paragraph(
                "This report was generated by an AI-assisted ECG monitoring system for "
                "research and screening purposes only. The artificial intelligence model "
                "(ECG-ResNet-SE) was trained on the MIT-BIH Arrhythmia Database and provides "
                "probabilistic beat classification results.",
                disclaimer_style,
            ),
            Paragraph(
                "THIS REPORT DOES NOT CONSTITUTE A MEDICAL DIAGNOSIS. All findings must be "
                "reviewed and confirmed by a qualified cardiologist or licensed physician before "
                "any clinical decisions are made. The AI system may produce incorrect "
                "classifications, particularly for rare beat morphologies or signals contaminated "
                "by motion artifacts.",
                disclaimer_style,
            ),
            Paragraph(
                "For research use only. Not approved for clinical diagnostic use. "
                f"Report generated by ECG Monitor v1.0 on {datetime.utcnow().strftime('%Y-%m-%d')}.",
                disclaimer_style,
            ),
        ]
