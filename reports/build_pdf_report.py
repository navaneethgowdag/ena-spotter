"""
Script to generate reports/Freight_Rate_Report.pdf using ReportLab.
Produces a comprehensive, professional ML assessment report with embedded charts and tables.
"""
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print total page numbers (e.g. Page 1 of 5).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4a5568"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Spotter ML Take-Home Assessment — Freight Rate Prediction Report")
            self.setStrokeColor(colors.HexColor("#cbd5e0"))
            self.setLineWidth(0.5)
            self.line(54, 747, 558, 747)

        # Footer
        self.setStrokeColor(colors.HexColor("#cbd5e0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        
        self.drawString(54, 32, "CONFIDENTIAL — Spotter Machine Learning Engineering Assessment")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()


def create_report(output_pdf_path: str):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom palette
    PRIMARY = colors.HexColor("#064A56")     # Spotter brand teal
    SECONDARY = colors.HexColor("#1b4965")   # Slate navy
    ACCENT = colors.HexColor("#d97706")      # Amber accent
    TEXT_DARK = colors.HexColor("#1a202c")   # Body text
    MUTED = colors.HexColor("#4a5568")       # Caption / subtitle
    BG_LIGHT = colors.HexColor("#f7fafc")    # Card / table bg
    BORDER = colors.HexColor("#e2e8f0")

    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=MUTED,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=MUTED,
        alignment=1, # Center
        spaceBefore=4,
        spaceAfter=10
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=PRIMARY
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Freight Rate Prediction — Engineering Report", title_style))
    story.append(Paragraph("Spotter Machine Learning Engineer Take-Home Assessment | Candidate: Navaneeth Gowda", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=12))

    # Executive Summary Card
    summary_text = (
        "<b>Executive Summary:</b> This report presents an end-to-end Machine Learning pipeline to predict spot "
        "freight load rates from shipment specifications (origin/destination cities, distance, equipment type, weight, "
        "and dispatch dates). Using 48,000 historical development loads (Jan–Oct 2025), we built a leak-free time-series "
        "validation framework, audited critical data quality anomalies, engineered Bayesian smoothed target encodings "
        "with spatial fallbacks, and developed an L1-regularized LightGBM model optimizing rate-per-mile ($/mi). On held-out "
        "future months (Sep–Oct 2025), our LightGBM model achieves an <b>MAE of $106.16</b> and <b>R² of 0.8281</b> (a 46% error reduction "
        "over the historical per-lane median baseline). The pipeline was deployed to generate compliant predictions for "
        "all 12,000 validation loads and the fixed 31-day December 2025 forecast (mean: $834.20), verified error-free by the official score.py."
    )
    summary_table = Table([[Paragraph(summary_text, callout_style)]], colWidths=[504])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#e6fffa")),
        ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor("#81e6d9")),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # Section 1: Data Exploration & Quality Audit
    story.append(Paragraph("1. Data Exploration & Quality Audit", h1_style))
    story.append(Paragraph(
        "A rigorous audit of <code>train_test.csv</code> (48,000 rows x 14 columns) and <code>validation.csv</code> "
        "(12,000 rows x 13 columns) uncovered key structural characteristics and four critical data-quality anomalies:",
        body_style
    ))

    data_issues = [
        "<b>Negative Weights (Sign Intake Anomaly):</b> 292 loads in train and 145 loads in validation contained negative "
        "weights (min: -47,500 lbs). Distribution analysis confirmed that <code>abs(weight)</code> has identical statistical "
        "properties to positive weights (mean: ~31,415 lbs). <i>Remediation:</i> Standardized via <code>np.abs(weight)</code>.",
        "<b>Missing Attribute Values:</b> 300 rows in train (0.62%) and 165 rows in validation (1.38%) lacked shipment weight. "
        "<i>Remediation:</i> Imputed with the development median (31,436.5 lbs) and encoded an explicit missing indicator flag.",
        "<b>Unseen Cities and Lanes in Validation:</b> While equipment types remained identical (Dry Van 55%, Reefer 25%, Flatbed 20%), "
        "validation introduced 8 new cities (e.g. Chicago, San Diego, Charlotte, Laredo) and <b>736 unseen origin-destination lanes (17.5% of validation loads)</b>. "
        "<i>Remediation:</i> Rather than relying on rigid categorical one-hot encoding, we constructed a hierarchical Bayesian smoothed "
        "target encoder that gracefully falls back from lane-level priors to origin/destination regional priors, equipment medians, and global medians, "
        "augmented with continuous geographic coordinates (pickup/delivery latitude and longitude).",
        "<b>Signal Leakage & Scorer Consistency Audit (<code>quote_signal</code> & <code>market_index</code>):</b> "
        "In <code>train_test.csv</code> and <code>validation.csv</code>, a feature named <code>quote_signal</code> is present. "
        "However, <code>december_chart_inputs.csv</code> <b>completely omits</b> <code>quote_signal</code> and <code>market_index</code>. "
        "Empirical benchmarking confirmed that a model trained purely on fundamental shipment attributes (distance, weight, equipment, coordinates, "
        "and calendar features) achieves superior generalization on held-out validation (<b>MAE $106.16</b> vs $108.33 for the model with quote_signal). "
        "Excluding quote_signal guarantees zero target leakage and ensures 100% architectural parity between validation and December forecasting."
    ]
    for issue in data_issues:
        story.append(Paragraph(f"• {issue}", bullet_style))

    story.append(Spacer(1, 8))

    # Embed Figure 1: Target Distribution
    fig1_path = "reports/figures/01_target_distribution.png"
    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=500, height=140))
        story.append(Paragraph("Figure 1: Target distribution of posted freight rates ($), log-transformed rates, and rate-per-mile ($/mi).", caption_style))

    story.append(Spacer(1, 6))

    # Section 2: Validation Strategy & Leakage Prevention
    story.append(Paragraph("2. Time-Based Validation Strategy", h1_style))
    story.append(Paragraph(
        "Freight rate prediction is an inherently temporal problem where models must price future contracts given past operating history. "
        "<b>Why random K-fold splits are invalid:</b> Random splits cause massive data leakage because loads dispatched on the same lane "
        "within the same week share identical macroeconomic freight cycles, diesel fuel surcharges, and spot market tightness. Evaluating on random "
        "hold-outs inflates performance metrics and severely misrepresents future generalization error.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Our Dual-Tier Validation Design:</b>",
        h2_style
    ))
    split_points = [
        "<b>Tier 1 — Held-Out Test Period (Sep 1 to Oct 31, 2025):</b> The final two months of development data (9,523 loads) were completely "
        "quarantined as a blind test set, perfectly replicating the task of predicting November and December validation loads.",
        "<b>Tier 2 — Forward-Chaining (Expanding-Window) Cross-Validation:</b> Inside the training partition (Jan 1 to Aug 31, 2025; 38,477 loads), "
        "we executed 4 expanding-window folds (Fold 1: train Jan–Apr -> validate May; Fold 2: train Jan–May -> validate Jun; Fold 3: train Jan–Jun -> "
        "validate Jul; Fold 4: train Jan–Jul -> validate Aug). Target encodings and aggregations were recomputed strictly inside each training window.",
        "<b>Final Model Refit:</b> After establishing hyperparameter stability across all 4 forward-chaining folds and evaluating all candidate "
        "architectures on the held-out September–October set, the selected LightGBM model was refit on all 48,000 loads (Jan–Oct) to produce the final submissions."
    ]
    for sp in split_points:
        story.append(Paragraph(f"• {sp}", bullet_style))

    story.append(Spacer(1, 10))

    # Section 3: Model Benchmarking & Performance Comparison
    story.append(Paragraph("3. Model Benchmarking & Metric Comparison", h1_style))
    story.append(Paragraph(
        "To rigorously quantify performance lift, five candidate models were trained on months 1–8 and evaluated on the held-out "
        "September–October test period (9,523 loads). Target formulation: We modeled <b>rate-per-mile ($/mi)</b> directly and multiplied by "
        "distance (<code>predicted_rate = predicted_rpm * distance</code>), naturally enforcing non-negativity and capturing haul-length economics.",
        body_style
    ))

    # Metrics Table
    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        alignment=1
    )
    cell_bold_style = ParagraphStyle(
        'TableCellBold',
        parent=cell_style,
        fontName='Helvetica-Bold',
        textColor=PRIMARY
    )
    cell_left_style = ParagraphStyle(
        'TableCellLeft',
        parent=cell_style,
        alignment=0
    )
    cell_left_bold_style = ParagraphStyle(
        'TableCellLeftBold',
        parent=cell_style,
        fontName='Helvetica-Bold',
        textColor=PRIMARY,
        alignment=0
    )

    table_data = [
        ["Model Architecture", "MAE ($)", "RMSE ($)", "MAPE (%)", "SMAPE (%)", "R² Score", "Key Characteristic"],
        ["Baseline: Global Median RPM", "$256.95", "$684.25", "11.63%", "11.02%", "0.7990", "Naive scalar heuristic"],
        ["Baseline: Per-Lane Median RPM", "$196.95", "$654.16", "8.58%", "7.82%", "0.8163", "Historical lane median fallback"],
        ["Ridge Linear Regression (log rate)", "$148.83", "$634.25", "6.58%", "5.47%", "0.8273", "L2 regularized linear model"],
        ["Random Forest Regressor (RPM target)", "$148.84", "$660.08", "6.47%", "5.37%", "0.8129", "Non-linear bagged decision trees"],
        [
            Paragraph("LightGBM Gradient Boosting (Winner)", cell_left_bold_style),
            Paragraph("$106.16", cell_bold_style),
            Paragraph("$632.66", cell_bold_style),
            Paragraph("4.62%", cell_bold_style),
            Paragraph("3.78%", cell_bold_style),
            Paragraph("0.8281", cell_bold_style),
            Paragraph("L1 loss, Bayesian priors, cyclic calendar", cell_left_bold_style)
        ]
    ]
    
    # Styled Table
    t = Table(table_data, colWidths=[150, 52, 54, 52, 54, 50, 92])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-2, -1), 'CENTER'),
        ('ALIGN', (-1, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, BG_LIGHT]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#e6fffa")), # Highlight winner
    ]))
    story.append(t)
    story.append(Paragraph("Table 1: Comprehensive benchmark comparison on the held-out September–October test partition.", caption_style))

    story.append(Spacer(1, 6))

    # Embed Figure 6: Model Comparison
    fig6_path = "reports/figures/06_model_comparison.png"
    if os.path.exists(fig6_path):
        story.append(Image(fig6_path, width=490, height=140))
        story.append(Paragraph("Figure 2: Performance comparison across candidate architectures on held-out test data.", caption_style))

    # Embed Figure 7: Feature Importance
    fig7_path = "reports/figures/07_feature_importance.png"
    if os.path.exists(fig7_path):
        story.append(Image(fig7_path, width=480, height=155))
        story.append(Paragraph("Figure 3: Feature importance (total information gain) in the production LightGBM model.", caption_style))

    story.append(Spacer(1, 8))

    # Section 4: December Forecast Analysis & Scorer Verification
    story.append(Paragraph("4. Fixed December 2025 Scenario & Scorer Verification", h1_style))
    story.append(Paragraph(
        "The assessment required predicting load rates for 31 consecutive days in December 2025 on a fixed lane: "
        "<b>Lexington to Fort Wayne, 360 miles, Dry Van equipment, 32,000 lb</b> (where only the date changes).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Interpretation of the Candidate December Chart:</b>",
        h2_style
    ))
    dec_points = [
        "<b>Historical Alignment:</b> In <code>train_test.csv</code>, historical Dry Van loads on Lexington -> Fort Wayne averaged 363.8 miles "
        "with an average rate of <b>$823.31</b> (median: $807.89). Our December predictions span from <b>$814.77 to $848.25</b> (mean: $834.20), "
        "showing remarkable fidelity to empirical ground truth.",
        "<b>Weekly Day-of-Week Rhythm:</b> The predicted curve demonstrates clear freight dispatch patterns: rates soften over the weekend "
        "(Sundays dip to ~$823–$828) and tighten midweek during peak shipper dispatch windows (Wednesdays and Thursdays rise to ~$840–$844).",
        "<b>Holiday Season Tightening:</b> Approaching Christmas, capacity tightens significantly, peaking on Christmas Day at <b>$848.25</b>, "
        "followed by post-holiday rebalancing and month-end settlement on Dec 31 ($843.05).",
        "<b>Smooth & Plausible Extrapolation:</b> Unlike naive tree models that produce erratic step functions when extrapolating beyond training dates, "
        "our model leverages cyclical calendar signals (sine/cosine day-of-week, proximity to holidays, and day of month), producing a smooth, defensible curve."
    ]
    for dp in dec_points:
        story.append(Paragraph(f"• {dp}", bullet_style))

    story.append(Spacer(1, 8))

    # Embed Candidate December Chart from score.py
    scorer_chart_path = "scorer_results/candidate_december.png"
    if os.path.exists(scorer_chart_path):
        story.append(Image(scorer_chart_path, width=490, height=170))
        story.append(Paragraph("Figure 4: Official Candidate December 2025 Predicted Load Rate chart generated by score.py.", caption_style))

    story.append(Spacer(1, 10))

    # Section 5: Production Execution & Reproducibility
    story.append(Paragraph("5. Code Architecture & Reproducibility", h1_style))
    story.append(Paragraph(
        "The project is structured according to professional software engineering standards. All experiments are 100% reproducible "
        "with fixed random seeds (SEED=42) and modular components in <code>src/</code>:",
        body_style
    ))
    
    code_summary = [
        "<code>src/config.py</code>: Centralized paths, fixed December scenario constants, US holidays, and LightGBM hyperparameters.",
        "<code>src/data_prep.py</code>: Clean dataset routines applied identically across train, validation, and December datasets.",
        "<code>src/features.py</code>: Bayesian smoothed target encoder, geographic coordinate lookups, and cyclical calendar transforms.",
        "<code>src/train.py</code>: Forward-chaining cross-validation runner and full-dataset production model trainer.",
        "<code>src/predict.py</code>: End-to-end inference engine generating <code>validation_predictions.csv</code>, updating "
        "<code>data/december_chart_inputs.csv</code>, and running automated <code>score.py</code> validation."
    ]
    for cs in code_summary:
        story.append(Paragraph(f"• {cs}", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Reproduction Command:</b><br/>"
        "<code>python -m src.train && python -m src.predict</code>",
        callout_style
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report generated successfully: {output_pdf_path}")


if __name__ == "__main__":
    create_report("reports/Freight_Rate_Report.pdf")
