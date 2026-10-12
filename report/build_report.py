"""Build the two-page environment assignment report."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "ME5418_Group18_Environment_Report.pdf"
DEMO_IMAGE = ROOT / "results" / "comparison" / "routes_seed_0.png"

INK = colors.HexColor("#16231D")
GREEN = colors.HexColor("#2F6B4F")
PALE_GREEN = colors.HexColor("#EAF3EE")
LIGHT_GRAY = colors.HexColor("#E4E7E5")
MID_GRAY = colors.HexColor("#59635E")


def font_family() -> tuple[str, str]:
    candidates = [
        (
            Path("C:/Windows/Fonts/aptos.ttf"),
            Path("C:/Windows/Fonts/aptosbd.ttf"),
            "Aptos",
            "Aptos-Bold",
        ),
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            "Arial",
            "Arial-Bold",
        ),
    ]
    for regular, bold, regular_name, bold_name in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont(regular_name, regular))
            pdfmetrics.registerFont(TTFont(bold_name, bold))
            return regular_name, bold_name
    return "Helvetica", "Helvetica-Bold"


REGULAR, BOLD = font_family()


def page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LIGHT_GRAY)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFillColor(MID_GRAY)
    canvas.setFont(REGULAR, 7.5)
    canvas.drawString(18 * mm, 8.5 * mm, "ME5418 Group 18 Environment Assignment")
    canvas.drawRightString(A4[0] - 18 * mm, 8.5 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    if not DEMO_IMAGE.exists():
        raise FileNotFoundError(f"Demo image not found: {DEMO_IMAGE}")
    paired_results = json.loads(
        (ROOT / "results" / "comparison" / "episodes.json").read_text(encoding="utf-8")
    )
    push = json.loads((ROOT / "results" / "demo" / "push_seed_0.json").read_text())
    detour = json.loads((ROOT / "results" / "demo" / "detour_seed_0.json").read_text())
    push_results = [item for item in paired_results if item["policy"] == "push"]
    detour_results = [item for item in paired_results if item["policy"] == "detour"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
        title="ME5418 Flexible Obstacle Interaction Environment Report",
        author="Group 18",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=BOLD,
        fontSize=17,
        leading=19,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=3 * mm,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName=REGULAR,
        fontSize=9,
        leading=11,
        textColor=MID_GRAY,
        spaceAfter=4 * mm,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName=BOLD,
        fontSize=11.8,
        leading=13.8,
        textColor=INK,
        spaceBefore=2.2 * mm,
        spaceAfter=1.4 * mm,
        keepWithNext=True,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=REGULAR,
        fontSize=9.4,
        leading=12.2,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=1.7 * mm,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=8.2,
        leading=10,
        textColor=MID_GRAY,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    cell = ParagraphStyle(
        "Cell",
        parent=body,
        fontSize=8.2,
        leading=10.2,
        spaceAfter=0,
    )
    cell_head = ParagraphStyle(
        "CellHead",
        parent=cell,
        fontName=BOLD,
        textColor=colors.white,
    )

    story = [
        Paragraph("Flexible Obstacle Interaction in Robotic Harvesting", title),
        Paragraph(
            "Environment implementation report &nbsp; | &nbsp; ME5418 Machine Learning in Robotics &nbsp; | &nbsp; Group 18",
            subtitle,
        ),
        Paragraph("1  Project overview", heading),
        Paragraph(
            "Our fully observed, two-dimensional Gymnasium environment models a circular robot end effector reaching fruit among flexible branches. Four branches form two opposing tip gaps. Each resting gap is narrower than the robot diameter, and its height varies between episodes. A policy can locate and gently open these internal passages or take a longer collision-free route around fixed anchors. Seeded geometry and bounded execution noise are configured separately.",
            body,
        ),
        Paragraph("2  Code explanation and organization", heading),
    ]

    table_data = [
        [Paragraph("Component", cell_head), Paragraph("Implementation", cell_head)],
        [
            Paragraph("Environment", cell),
            Paragraph(
                "<b>HarvestingEnv</b> implements Gymnasium <i>reset</i>, <i>step</i>, <i>render</i>, and <i>close</i>. It maintains episode state, applies one of eight planar actions, computes rewards, and returns termination diagnostics.",
                cell,
            ),
        ],
        [
            Paragraph("Scene and physics", cell),
            Paragraph(
                "<b>scene_generator.py</b> creates two tip gates with alternating boundary and interior anchors. <b>branch_physics.py</b> projects normal contact and segment-length constraints. Anchors stay fixed; stiffness limits node movement, and springs restore the resting shape after release.",
                cell,
            ),
        ],
        [
            Paragraph("Observation and action", cell),
            Paragraph(
                "The 125-dimensional default observation contains robot/target positions, five padded branch slots with current/resting nodes, stiffness, contact, safe limits, masks, and remaining budget. Eight actions move in four directions by 0.025 or 0.060.",
                cell,
            ),
        ],
        [
            Paragraph("Reward and episode end", cell),
            Paragraph(
                "Reward combines progress, travel and deformation costs, a damage penalty, and a safe-success bonus. Peak deformation is checked inside every substep. Success or damage terminates an episode; 100 actions cause truncation. A* and the demo controller use identical low-level actions.",
                cell,
            ),
        ],
    ]
    code_table = Table(table_data, colWidths=[39 * mm, 132 * mm], repeatRows=1)
    code_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GREEN),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#D2D8D5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("BACKGROUND", (0, 2), (-1, 2), PALE_GREEN),
                ("BACKGROUND", (0, 4), (-1, 4), PALE_GREEN),
            ]
        )
    )
    story.extend(
        [
            code_table,
            Spacer(1, 2.5 * mm),
            KeepTogether(
                [
                    Image(str(DEMO_IMAGE), width=171 * mm, height=92 * mm),
                    Spacer(1, 1 * mm),
                    Paragraph(
                        "Figure 1. Same scene, seed 0. Left: the contact controller follows the internal gaps; the inset shows the first opening. Right: A* keeps branch geometry unchanged and travels around the alternating anchors.",
                        small,
                    ),
                ]
            ),
            PageBreak(),
            Paragraph("3  Demonstration and reproducibility", heading),
            Paragraph(
                f"The graphical demo saves GIFs, initial/contact/final snapshots, state traces, and JSON metrics. A zoom panel makes tip separation visible. On seed 0, the contact controller succeeded in {push['steps']} actions, travelled {push['travel_distance']:.3f}, and reached peak normalized deformation {push['peak_normalized_deformation']:.3f}. A* succeeded in {detour['steps']} actions with travel {detour['travel_distance']:.3f} and zero contact. Across seeds 0-9, both succeeded in all ten scenes; mean travel was {mean(item['travel_distance'] for item in push_results):.3f} versus {mean(item['travel_distance'] for item in detour_results):.3f}. These are environment diagnostics, not learned-policy evaluation.",
                body,
            ),
            Paragraph("Reproduction commands", heading),
        ]
    )

    commands = Table(
        [
            [Paragraph("Create environment", cell_head), Paragraph("conda env create -f environment.yml", cell)],
            [Paragraph("Activate", cell_head), Paragraph("conda activate me5418-harvesting", cell)],
            [Paragraph("Contact demo", cell_head), Paragraph("python scripts/run_demo.py --policy push --seed 0", cell)],
            [Paragraph("Same-scene detour", cell_head), Paragraph("python scripts/run_demo.py --policy detour --seed 0", cell)],
            [Paragraph("Run tests", cell_head), Paragraph("pytest -q", cell)],
        ],
        colWidths=[43 * mm, 128 * mm],
    )
    commands.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), GREEN),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#D2D8D5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("BACKGROUND", (1, 1), (1, 1), PALE_GREEN),
                ("BACKGROUND", (1, 3), (1, 3), PALE_GREEN),
            ]
        )
    )
    story.extend(
        [
            commands,
            Paragraph("4  Existing code, libraries, and tools", heading),
            Paragraph(
                "Task geometry, contact projections, rewards, A* search, and demonstration controllers are implemented in this repository. Gymnasium supplies the environment interface; NumPy handles geometry; Matplotlib and Pillow produce graphics; PyYAML loads configuration; pytest and Ruff support checks. No external robotics simulator is required. Codex assisted implementation, debugging, and documentation. Optional PyTorch code currently defines the DQN network; the training loop remains future work.",
                body,
            ),
            Paragraph("5  Reflections and lessons learned", heading),
            Paragraph(
                "Parallel branches with a shared top clearance made the original task an outer-detour problem. Opposing tips create a measurable contact opportunity: gaps of 0.028-0.043 are smaller than the robot diameter of 0.070. Random gap heights require alignment. Alternating interior anchors preserve a rigid-geometry route without a universal top or bottom shortcut. Motion is subdivided, difficult substeps are reduced, and contact/segment constraints are checked together to avoid tunnelling and numerical sticking.",
                body,
            ),
            Paragraph(
                "Nine tests cover interface/reproducibility/rendering, gap geometry, rigid blocking, soft opening, fixed anchors, segment strain, and recovery. The geometric surrogate allows 3 percent segment strain and omits inertia, hysteresis, permanent deformation, and inter-branch mechanics. Its normalized parameters require physical calibration. One checked contact episode approaches its damage limit; ten successful seeds do not establish robustness. The contact controller is handwritten, and the default reward still prefers zero-contact detours despite their longer travel.",
                body,
            ),
            Paragraph("6  Remaining work toward the final project", heading),
            Paragraph(
                "Final work will implement DQN training and compare it with collision-free, myopic, and receding-horizon methods on held-out seeds. Gap width, stiffness, safety limits, reward weights, and execution noise need sensitivity tests. We will report success/damage, actions, travel, peak deformation, return, model evaluations, decision time, and training cost. Evidence of an RL advantage must come from that comparison, rather than from the scripted contact demonstration.",
                body,
            ),
        ]
    )

    document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(OUTPUT)


if __name__ == "__main__":
    build()
