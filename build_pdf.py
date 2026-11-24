"""Build Adversarial_Robustness_Documentation.pdf: documentation of the whole code base + results.

    python build_pdf.py [results_dir] [runs_dir]        (defaults: results_cpu_demo runs_cpu_demo)

Source listings are read from the real files, results from results.json / history.json / PNG figures.
"""
import json
import os
import re
import sys
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus.flowables import Flowable, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

RES = sys.argv[1] if len(sys.argv) > 1 else "results_cpu_demo"
RUNS = sys.argv[2] if len(sys.argv) > 2 else "runs_cpu_demo"
OUT = "Adversarial_Robustness_Documentation.pdf"
NAVY, TEAL = colors.HexColor("#14213d"), colors.HexColor("#0f9d8a")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=18, leading=22, textColor=NAVY, spaceBefore=4, spaceAfter=2)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, leading=16, textColor=TEAL, spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.8, leading=13.5, textColor=NAVY, spaceBefore=12, spaceAfter=5)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.8, leading=14.6, spaceAfter=7, alignment=TA_JUSTIFY)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.2, leading=10.5, textColor=colors.HexColor("#555555"))
CODE = ParagraphStyle("C", fontName="Courier", fontSize=7, leading=8.7)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.6, leading=11, spaceAfter=0)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=26, leading=31, textColor=NAVY, alignment=TA_CENTER)
SUB = ParagraphStyle("Sub", parent=BODY, fontSize=12, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))

ASCII = {"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "≤": "<=",
         "×": "x", "→": "->", "ε": "eps", "α": "alpha", "δ": "delta", "≈": "~"}


def clean(s):
    for k, v in ASCII.items():
        s = s.replace(k, v)
    return s.encode("ascii", "replace").decode()


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [P("&bull;&nbsp; " + i, ParagraphStyle("bl", parent=BODY, leftIndent=14, firstLineIndent=-10, spaceAfter=4, alignment=0))
            for i in items]


class CodeBox(Flowable):
    """Monospaced code box: every line is drawn at an exact column, so indentation is preserved."""
    FS, LEAD, PAD, GUT = 7.0, 8.8, 7, 20

    def __init__(self, rows, width):
        super().__init__()
        self.rows, self.width = rows, width

    def wrap(self, aw, ah):
        self.height = len(self.rows) * self.LEAD + 2 * self.PAD
        return self.width, self.height

    @staticmethod
    def _split_comment(text):
        q = None
        for i, ch in enumerate(text):
            if q:
                if ch == q:
                    q = None
            elif ch in "\"'":
                q = ch
            elif ch == "#":
                return text[:i], text[i:]
        return text, ""

    def draw(self):
        c, w, h = self.canv, self.width, self.height
        c.setFillColor(colors.HexColor("#f6f7fa")); c.setStrokeColor(colors.HexColor("#9aa3b2")); c.setLineWidth(0.8)
        c.rect(0, 0, w, h, stroke=1, fill=1)
        c.setFillColor(TEAL); c.rect(0, 0, 3, h, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#eceff4")); c.rect(3, 0, self.GUT + 2, h, stroke=0, fill=1)
        cw = pdfmetrics.stringWidth("M", "Courier", self.FS)
        x0 = 3 + self.GUT + 2 + self.PAD - 2
        for i, (no, text) in enumerate(self.rows):
            y = h - self.PAD - (i + 1) * self.LEAD + 2.4
            if no:
                c.setFont("Courier", self.FS - 0.8); c.setFillColor(colors.HexColor("#8a93a3"))
                c.drawRightString(3 + self.GUT, y, str(no))
            code, cmt = self._split_comment(text)
            c.setFont("Courier", self.FS); c.setFillColor(colors.HexColor("#1b1f2a"))
            c.drawString(x0, y, code)
            if cmt:
                c.setFillColor(colors.HexColor("#5b8a72")); c.drawString(x0 + cw * len(code), y, cmt)


def code_block(text, width=96, chunk=26):
    rows = []
    for no, ln in enumerate(clean(text).expandtabs(4).rstrip().split(chr(10)), 1):
        ind = len(ln) - len(ln.lstrip())
        pieces = textwrap.wrap(ln, width, subsequent_indent=" " * (ind + 4), drop_whitespace=False,
                               replace_whitespace=False) or [""]
        rows += [(no if k == 0 else "", pc.rstrip()) for k, pc in enumerate(pieces)]
    out = []
    for i in range(0, len(rows), chunk):                       # chunks so long listings break across pages
        out += [CodeBox(rows[i:i + chunk], 166 * mm), Spacer(1, 8)]
    return out


def table(rows, widths, header=True, font=8.6):
    data = [[Paragraph(str(c), ParagraphStyle("tc", parent=CELL, fontSize=font, leading=font + 2.4,
                                              textColor=colors.white if (header and i == 0) else colors.black))
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ccd4")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
          ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
          ("BOX", (0, 0), (-1, -1), 0.9, NAVY)]
    for r in range(1 if header else 0, len(data)):
        if (r % 2) == 0:
            st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f1f4f8")))
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    t.setStyle(TableStyle(st))
    t.hAlign = 'CENTER'
    return t


def read(path):
    return open(path, encoding="utf-8").read()


def figure(path, caption, width=150 * mm):
    if not os.path.exists(path):
        return []
    from PIL import Image as PI
    w, h = PI.open(path).size
    return [KeepTogether([Image(path, width=width, height=width * h / w), P(caption, SMALL)])]


def _frame(c):
    w, h = A4
    c.setStrokeColor(NAVY); c.setLineWidth(1.6)
    c.rect(9 * mm, 9 * mm, w - 18 * mm, h - 18 * mm)
    c.setStrokeColor(TEAL); c.setLineWidth(0.5)
    c.rect(11.2 * mm, 11.2 * mm, w - 22.4 * mm, h - 22.4 * mm)


def cover(c, d):
    w, h = A4
    c.saveState()
    m = 9 * mm
    c.setFillColor(NAVY); c.rect(m, h - m - 56 * mm, w - 2 * m, 56 * mm, stroke=0, fill=1)
    c.setFillColor(TEAL); c.rect(m, h - m - 60 * mm, w - 2 * m, 4 * mm, stroke=0, fill=1)
    c.setFillColor(NAVY); c.rect(m, m, w - 2 * m, 20 * mm, stroke=0, fill=1)
    c.setFillColor(TEAL); c.rect(m, m + 20 * mm, w - 2 * m, 2.5 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(w / 2, h - m - 26 * mm, "DEEP LEARNING (BCS714A)  |  ACTIVITY-BASED LEARNING")
    c.setFont("Helvetica", 9.5)
    c.drawCentredString(w / 2, m + 8 * mm, "Academic Year 2026-27")
    _frame(c)
    c.restoreState()


def footer(c, d):
    w, h = A4
    c.saveState()
    _frame(c)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawString(20 * mm, h - 17 * mm, "Adversarial Attacks and Robustness")
    c.drawRightString(w - 20 * mm, h - 17 * mm, "Code Documentation and Results")
    c.setStrokeColor(colors.HexColor("#c8ccd4")); c.setLineWidth(0.4)
    c.line(20 * mm, h - 19 * mm, w - 20 * mm, h - 19 * mm)
    c.line(20 * mm, 17 * mm, w - 20 * mm, 17 * mm)
    c.drawCentredString(w / 2, 13.2 * mm, f"- {d.page} -")
    c.restoreState()


story = []

# ---------------------------------------------------------------- title page
story += [Spacer(1, 52 * mm), P("Adversarial Attacks and Robustness", TITLE), Spacer(1, 6),
          P("CIFAR-10 ResNet-18 &nbsp;|&nbsp; FGSM, PGD and PGD Adversarial Training", SUB), Spacer(1, 10 * mm),
          P("<b>Complete Code Documentation and Results</b>", SUB), Spacer(1, 18 * mm),
          P("Deep Learning (BCS714A), VII Semester B.E. - Activity-Based Learning", SUB),
          P("Department of CSE (Data Science), Sri Venkateshwara College of Engineering, Bengaluru", SUB),
          Spacer(1, 8 * mm), P("Group G20: R K Shravanya (1VE23CD040) &nbsp;&amp;&nbsp; Rakesh Behera (1VE23CD041)", SUB),
          P("Guide: Prof. Sruthi Krishna U", SUB), PageBreak()]

# ---------------------------------------------------------------- 1 overview
story += [P("1. Project Overview", H1),
          P("Deep neural networks reach very high accuracy on image classification but can be fooled by tiny, "
            "carefully crafted perturbations that humans cannot see. Such inputs are called <i>adversarial examples</i>. "
            "This project implements, from scratch in PyTorch, an attack-and-defense pipeline on CIFAR-10: two "
            "gradient-based attacks (FGSM and PGD) and one defense (PGD adversarial training), and measures how much "
            "robustness the defense buys and what it costs."),
          P("1.1 Objectives", H2)]
story += bullets([
    "Understand adversarial vulnerability under the L-infinity threat model (||delta||<sub>inf</sub> &lt;= eps).",
    "Implement FGSM (single step) and PGD (iterative, random start, projection) using <font face='Courier'>torch.autograd.grad</font>.",
    "Train a ResNet-18 two ways: standard training and PGD adversarial training (Madry et al. min-max objective).",
    "Evaluate clean, FGSM and PGD-20 accuracy and quantify the robustness/accuracy trade-off and training cost.",
    "Evaluate honestly: strong iterative attacks, fresh test examples, and an epsilon sweep."])
story += [P("1.2 Experimental settings", H2),
          table([["Item", "Setting"],
                 ["Dataset", "CIFAR-10: 50,000 training / 10,000 test images, 10 classes, 32x32 RGB"],
                 ["Augmentation", "random crop (padding 4) and horizontal flip (training only)"],
                 ["Model", "ResNet-18 for 32x32 inputs: 3x3 stem, no max-pool, 11,173,962 parameters"],
                 ["Threat model", "L-infinity, eps = 8/255, PGD step alpha = 2/255"],
                 ["Training attack", "PGD with random start; K = 10 steps (full experiment)"],
                 ["Evaluation", "clean, FGSM (1 step) and PGD-20 with random start, on test images"],
                 ["Optimiser", "SGD, momentum 0.9 (Nesterov), weight decay 5e-4, one-cycle LR schedule (max 0.1)"],
                 ["Framework", "PyTorch / torchvision, optional mixed precision on CUDA"]],
                [38 * mm, 136 * mm]), PageBreak()]

# ---------------------------------------------------------------- 2 theory
story += [P("2. Method", H1),
          P("2.1 FGSM", H2),
          P("For a classifier f with loss L, image x in [0,1]<super>n</super> and label y, FGSM linearises the loss "
            "around x and moves every pixel by eps in the direction that increases the loss:"),
          P("<font face='Courier'>x_adv = clip( x + eps * sign( grad_x L(f(x), y) ), 0, 1 )</font>"),
          P("It needs one backward pass, so it is cheap, but it is a weak attack: a single linear step often "
            "misses the worst case inside the eps-ball."),
          P("2.2 PGD", H2),
          P("PGD starts at a random point inside the eps-ball and repeats K gradient-sign steps of size alpha. "
            "After every step it projects the perturbation back to [-eps, eps] and clips the image to [0,1]:"),
          P("<font face='Courier'>x0 = x + U(-eps, eps);&nbsp; x(t+1) = Proj_eps( x(t) + alpha * sign( grad_x L ) )</font>"),
          P("PGD is regarded as the strongest first-order attack, which is why it is used both inside training and "
            "for the final evaluation (K = 20 at test time)."),
          P("2.3 Adversarial training (min-max)", H2),
          P("<font face='Courier'>min_theta E[ max_{||delta||inf &lt;= eps} L( f_theta(x + delta), y ) ]</font>"),
          P("The inner maximisation is solved approximately by PGD for every mini-batch; the outer minimisation is an "
            "ordinary SGD step on those worst-case inputs. Each batch costs about K+1 forward/backward passes, which is "
            "the source of the large training-time overhead. This project also provides <b>fast adversarial training</b> "
            "(<font face='Courier'>fast_at</font>): FGSM with a random start and step 1.25*eps, about ten times cheaper."),
          P("2.4 Pipeline", H2),
          table([["Stage", "What it does", "Code"],
                 ["Data", "load CIFAR-10, augment, keep pixels in [0,1]", "get_loaders"],
                 ["Model", "ResNet-18; mean/std normalisation inside the network", "ResNet18"],
                 ["Attack", "FGSM / PGD with eps-ball projection", "fgsm_attack, pgd_attack"],
                 ["Train", "standard, PGD-AT or fast-AT", "train_epoch, fit"],
                 ["Evaluate", "clean / FGSM / PGD-20, epsilon sweep, plots", "evaluate, eps_sweep, plot_*"]],
                [22 * mm, 96 * mm, 56 * mm]), PageBreak()]

# ---------------------------------------------------------------- 3 structure
story += [P("3. Repository Structure and Setup", H1),
          table([["File", "Purpose"],
                 ["advrob.py", "The whole implementation: data, ResNet-18, FGSM, PGD, training, evaluation, plots"],
                 ["train.py", "Command-line training of one model (standard / pgd_at / fast_at)"],
                 ["evaluate.py", "Evaluates trained models; writes the results table, JSON and figures"],
                 ["Adversarial_Robustness_Colab.ipynb", "Google Colab notebook (T4 GPU) generated from advrob.py"],
                 ["make_notebook.py", "Rebuilds the notebook from advrob.py so the two never differ"],
                 ["prepare_data.py", "Optional: fetch CIFAR-10 from a mirror and cache it as data/cifar10.npz"],
                 ["run_cpu_demo.ps1", "Reduced-scale run that finishes in about an hour on a laptop CPU"],
                 ["build_pdf.py", "Generates this document from the sources and result files"],
                 ["requirements.txt, README.md", "Dependencies and usage instructions"]],
                [58 * mm, 116 * mm]),
          P("3.1 Environment", H2)]
story += bullets([
    "Python 3.11, PyTorch, torchvision, NumPy, Matplotlib (see requirements.txt).",
    "The development machine has no NVIDIA GPU (Intel Core i5-13500H CPU). The installed system PyTorch failed to load "
    "because of an outdated Visual C++ runtime, so a project-local virtual environment (.venv) with the "
    "<font face='Courier'>msvc-runtime</font> package was used.",
    "The official CIFAR-10 download host was very slow, so <font face='Courier'>prepare_data.py</font> fetches the same "
    "dataset from a Hugging Face mirror; <font face='Courier'>get_loaders</font> uses the cached copy when present."])
story += [P("3.2 Commands", H2)]
story += code_block("""python -m venv .venv
.venv\\Scripts\\python -m pip install -r requirements.txt

python train.py --mode standard --epochs 20 --amp      # baseline
python train.py --mode pgd_at   --epochs 25 --amp      # PGD-10 adversarial training
python evaluate.py                                      # table + figures in results/

powershell -File run_cpu_demo.ps1                       # reduced run for a CPU-only laptop""")
story += [PageBreak()]

# ---------------------------------------------------------------- 4 code
story += [P("4. Code Walkthrough", H1),
          P("This section documents every source file. Listings are the actual files from the project.")]

adv = read("advrob.py")
parts = re.split(r"^# ===== SECTION: (.*) =====\n", adv, flags=re.M)
sections = list(zip(parts[1::2], parts[2::2]))
NOTES = {
    1: ["<b>Constants and helpers.</b> EPS = 8/255 and ALPHA = 2/255 define the threat model. "
        "<font face='Courier'>get_device</font> picks CUDA if available, <font face='Courier'>set_seed</font> makes runs repeatable."],
    2: ["<b>Data.</b> Images stay in [0,1] so that eps is measured in real pixel units; mean/std normalisation is "
        "done inside the model. Training uses random crop + flip, test images are unmodified. "
        "<font face='Courier'>train_subset</font>/<font face='Courier'>test_subset</font> allow small CPU runs, and "
        "<font face='Courier'>CIFAR10Arrays</font> reads the cached npz when it exists."],
    3: ["<b>Model.</b> ResNet-18 adapted to 32x32 inputs: a 3x3 stride-1 stem and no max-pool keep spatial resolution. "
        "Four stages of two BasicBlocks (64-128-256-512 channels), global average pooling and a linear classifier give "
        "11,173,962 parameters. Normalisation is applied in <font face='Courier'>forward</font>, so attacks work directly on [0,1] images."],
    4: ["<b>Attacks.</b> <font face='Courier'>_input_grad</font> returns the gradient of the cross-entropy loss with respect to "
        "the <i>input</i> using <font face='Courier'>torch.autograd.grad</font> (no weight gradients are accumulated). "
        "The loss uses reduction='sum' so fp16 gradients do not underflow, and NaNs are removed.",
        "<font face='Courier'>fgsm_attack</font> takes one signed step. <font face='Courier'>pgd_attack</font> starts from a random "
        "point, takes <i>steps</i> signed steps, and after every step clamps delta to [-eps, eps] and the image to [0,1]. "
        "Both switch the model to eval mode while attacking (fixed BatchNorm statistics) and restore the previous mode."],
    5: ["<b>Training.</b> <font face='Courier'>train_epoch</font> implements the three modes. In pgd_at the inner loop crafts the "
        "worst-case batch with PGD and the outer step updates the weights on it (Madry min-max). fast_at uses one FGSM step "
        "with random start and step 1.25*eps. <font face='Courier'>fit</font> builds a fresh ResNet-18 with SGD + one-cycle "
        "schedule, logs every epoch and measures clean and PGD accuracy on a small monitor subset of the test set."],
    6: ["<b>Evaluation.</b> <font face='Courier'>evaluate</font> returns clean, FGSM and PGD accuracy. PGD uses a random start "
        "and adversarial examples are generated fresh from test images, never reused from training. "
        "<font face='Courier'>eps_sweep</font> repeats PGD for growing budgets; save/load helpers store checkpoints with history."],
    7: ["<b>Plots.</b> Accuracy bars, adversarial examples (original / PGD-20 / amplified perturbation), robustness versus "
        "epsilon, and training curves."]}
story += [P("4.1 advrob.py", H2)]
for i, (title, body) in enumerate(sections, 1):
    story += [P(f"Section {title}", H3)] + [P(n) for n in NOTES.get(i, [])] + code_block(body)

for name, desc in [("train.py", "Command-line entry point for training. It parses the options, builds the loaders, calls "
                                "<font face='Courier'>fit</font>, and saves <font face='Courier'>model.pt</font> (weights, history, "
                                "settings) and <font face='Courier'>history.json</font>."),
                   ("evaluate.py", "Loads every trained model found in the runs folder, computes clean / FGSM / PGD-20 accuracy, "
                                   "the epsilon sweep and adversarial-example figures, and writes a markdown table that mirrors "
                                   "the results table of the report, plus <font face='Courier'>results.json</font>."),
                   ("prepare_data.py", "Downloads the CIFAR-10 parquet files from a mirror and caches uint8 arrays as "
                                       "<font face='Courier'>data/cifar10.npz</font>."),
                   ("make_notebook.py", "Splits advrob.py at the SECTION markers and writes each section as a notebook cell, then "
                                        "adds configuration, training, evaluation and download cells. The notebook therefore always "
                                        "contains exactly the tested code."),
                   ("run_cpu_demo.ps1", "Reduced-scale run for a CPU-only machine.")]:
    story += [P(f"4.{['train.py','evaluate.py','prepare_data.py','make_notebook.py','run_cpu_demo.ps1'].index(name)+2} {name}", H2), P(desc)]
    story += code_block(read(name))
story += [PageBreak()]

# ---------------------------------------------------------------- 5 verification
story += [P("5. Verification Performed", H1),
          table([["Check", "Result"],
                 ["Parameter count of ResNet-18 (CIFAR variant)", "11.17 M, matches the report"],
                 ["FGSM and PGD stay inside the eps-ball", "max |delta| = 8.000/255 for both attacks"],
                 ["Adversarial images remain valid", "all pixels within [0, 1]"],
                 ["Model train/eval mode restored after attacking", "yes"],
                 ["All three training modes and evaluate.py", "ran end to end on a tiny subset"],
                 ["Colab notebook", "all 13 code cells executed successfully in QUICK mode"]],
                [95 * mm, 79 * mm]),
          P("The notebook was executed locally on CPU in QUICK mode only. It has not yet been run on a Colab GPU.", SMALL)]

# ---------------------------------------------------------------- 6 results
story += [P("6. Results", H1)]
rj = os.path.join(RES, "results.json")
if os.path.exists(rj):
    R = json.load(open(rj))
    hist = {}
    for k, label in (("standard", "Standard training"), ("pgd_at", "PGD adversarial training")):
        hp = os.path.join(RUNS, k, "history.json")
        if os.path.exists(hp):
            hist[label] = json.load(open(hp))
    story += [P("6.1 Reduced-scale CPU run (measured)", H2)]
    cfg = {n: h["meta"] for n, h in hist.items()}
    story += [P("These are <b>real measurements</b> from the run_cpu_demo.ps1 configuration, executed on a laptop CPU. "
                "To make this feasible the models were trained on only <b>4,000 training images</b> (standard: 10 epochs; "
                "PGD adversarial training: 6 epochs with 3-step PGD instead of 10) and evaluated on a random subset of "
                f"<b>{R['n_test']} test images</b> with PGD-{R['pgd_steps']}. Absolute accuracies are therefore far below those of "
                "the full-scale experiment; the purpose is to show the qualitative effect of adversarial training.")]
    rows = [["Model", "Clean", "FGSM", f"PGD-{R['pgd_steps']}", "Time / epoch"]]
    for n, r in R["results"].items():
        rows.append([n, f"{r['clean']:.1f}%", f"{r['fgsm']:.1f}%", f"{r['pgd']:.1f}%",
                     f"{R['train_minutes'][n] * 60 / R['epochs'][n]:.0f} s"])
    story += [table(rows, [62 * mm, 24 * mm, 24 * mm, 24 * mm, 36 * mm]), Spacer(1, 6)]
    s, a = R["results"].get("Standard training"), R["results"].get("PGD adversarial training")
    if s and a:
        ov = (R["train_minutes"]["PGD adversarial training"] / R["epochs"]["PGD adversarial training"]) / \
             (R["train_minutes"]["Standard training"] / R["epochs"]["Standard training"])
        story += [P(f"Change from standard to PGD-adversarial training: clean accuracy {a['clean'] - s['clean']:+.1f} points, "
                    f"PGD-{R['pgd_steps']} accuracy {a['pgd'] - s['pgd']:+.1f} points, training time per epoch {ov:.1f}x "
                    "(3-step PGD; 10-step PGD costs roughly three times more per epoch).")]
    story += figure(os.path.join(RES, "accuracy_bars.png"), "Figure 1. Accuracy under attack (reduced CPU run).")
    story += figure(os.path.join(RES, "eps_sweep.png"), "Figure 2. PGD accuracy versus attack budget eps (reduced CPU run).", 125 * mm)
    story += figure(os.path.join(RES, "training_curves.png"), "Figure 3. Per-epoch monitor accuracy (small test subset, so noisy).")
    for key, cap in (("standard", "Figure 4. PGD-20 adversarial examples on the standard model."),
                     ("pgd_at", "Figure 5. PGD-20 adversarial examples on the adversarially trained model.")):
        story += figure(os.path.join(RES, f"adversarial_examples_{key}.png"), cap, 140 * mm)
    story += [P("6.2 Training logs", H2)]
    for n, h in hist.items():
        rows = [["Epoch", "Loss", "Train acc", "Test clean*", "Test PGD*", "Seconds"]]
        rows += [[r["epoch"], f"{r['loss']:.3f}", f"{r['train_acc']:.1f}%", f"{r['test_clean']:.1f}%",
                  f"{r['test_pgd']:.1f}%", f"{r['seconds']:.0f}"] for r in h["history"]]
        story += [P(n, H3), table(rows, [18 * mm, 24 * mm, 28 * mm, 30 * mm, 30 * mm, 24 * mm], font=8)]
    story += [P("* measured on a small monitor subset of the test set after each epoch (noisy); the table in 6.1 is the "
                "final evaluation.", SMALL)]
else:
    story += [P("Result files were not found; run run_cpu_demo.ps1 and rebuild this document.")]

story += [P("6.3 Full-scale experiment", H2),
          P("The full experiment (50,000 images, ResNet-18, PGD-10 training, 20-25 epochs) needs a GPU and is run with the "
            "Colab notebook. <b>It has not been run yet, so no full-scale numbers are reported here.</b> The reference values "
            "quoted in the project report (about 94% / 84% clean and 0% / 48% PGD-20 for standard / adversarial training) "
            "come from published CIFAR-10 benchmarks, not from this code base; after the Colab run they must be replaced by "
            "the measured values.")]

# ---------------------------------------------------------------- 7 discussion
story += [P("7. Observations, Limitations and Future Work", H1)]
story += bullets([
    "Robustness depends on the threat model: a model trained against eps = 8/255 L-infinity perturbations can still fail for "
    "larger budgets (see the epsilon sweep), other norms, or unseen attack types.",
    "Weak attacks overestimate robustness: FGSM accuracy is higher than PGD accuracy, so evaluation must use strong iterative attacks.",
    "Robustness has a price: lower clean accuracy and many times the training compute (about K+1 passes per batch).",
    "Not evaluated here: C&amp;W and AutoAttack, black-box/transfer attacks, certified defenses (randomised smoothing), "
    "other norms, patch/physical attacks.",
    "Future work: fast adversarial training at scale (fast_at is already implemented), AutoAttack evaluation, certified "
    "robustness, and robustness of vision transformers (cf. the computational-pathology study that motivated this topic)."])
story += [P("References", H2)]
story += [P(r, SMALL) for r in [
    "[1] Goodfellow, Shlens, Szegedy. Explaining and Harnessing Adversarial Examples. ICLR 2015.",
    "[2] Madry, Makelov, Schmidt, Tsipras, Vladu. Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018.",
    "[3] Carlini, Wagner. Towards Evaluating the Robustness of Neural Networks. IEEE S&amp;P 2017.",
    "[4] Szegedy et al. Intriguing Properties of Neural Networks. ICLR 2014.",
    "[5] Cohen, Rosenfeld, Kolter. Certified Adversarial Robustness via Randomized Smoothing. ICML 2019.",
    "[6] Athalye, Carlini, Wagner. Obfuscated Gradients Give a False Sense of Security. ICML 2018.",
    "[7] Wong, Rice, Kolter. Fast is Better than Free: Revisiting Adversarial Training. ICLR 2020.",
    "[8] Ghaffari Laleh et al. Adversarial attacks and adversarial robustness in computational pathology. Nature Communications 13, 2022."]]

# rule under every H1 heading
final = []
for f in story:
    final.append(f)
    if isinstance(f, Paragraph) and f.style.name == "H1":
        final.append(HRFlowable(width="100%", thickness=1.6, color=TEAL, spaceBefore=1, spaceAfter=10))
story = final

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=26 * mm, bottomMargin=24 * mm,
                        title="Adversarial Attacks and Robustness - Code Documentation", author="Group G20")
doc.build(story, onFirstPage=cover, onLaterPages=footer)
print("wrote", OUT)
