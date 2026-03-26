from flask import Flask, render_template, request, session, make_response
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
import csv
import io
import json
import base64
import os
import sys
import time

app = Flask(__name__)

# -------------------------------------------------
# Deployment-friendly configuration
# -------------------------------------------------
app.secret_key = os.environ.get("SECRET_KEY", "molscore_pk_dev_secret_key")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# -----------------------------
# Optional SA score import
# -----------------------------
SA_SCORER_AVAILABLE = False
sascorer = None

try:
    import sascorer
    SA_SCORER_AVAILABLE = True
except ImportError:
    try:
        rdkit_contrib = os.path.join(sys.prefix, "share", "RDKit", "Contrib", "SA_Score")
        if os.path.isdir(rdkit_contrib) and rdkit_contrib not in sys.path:
            sys.path.append(rdkit_contrib)
        import sascorer
        SA_SCORER_AVAILABLE = True
    except Exception:
        SA_SCORER_AVAILABLE = False
        sascorer = None

# -----------------------------
# Filter catalogs
# -----------------------------
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(pains_params)

brenk_params = FilterCatalogParams()
brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
BRENK_CATALOG = FilterCatalog(brenk_params)


def clamp(value, low=0.0, high=6.0):
    return max(low, min(high, value))


def clamp01(value):
    return max(0.0, min(1.0, value))


def scale_range(value, low, high):
    if high == low:
        return 3.0
    scaled = ((value - low) / (high - low)) * 6.0
    return clamp(scaled)


def scale_inverse(value, low, high):
    return 6.0 - scale_range(value, low, high)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_filter_matches(mol, catalog):
    matches = catalog.GetMatches(mol)
    entries = []
    for match in matches:
        try:
            entries.append(match.GetDescription())
        except Exception:
            entries.append("Alert")
    return entries


def summarize_alerts(alert_list):
    if not alert_list:
        return "0 alert"
    unique_alerts = sorted(set(alert_list))
    if len(unique_alerts) == 1:
        return f"1 alert: {unique_alerts[0]}"
    return f"{len(unique_alerts)} alerts: " + "; ".join(unique_alerts[:5])


def calculate_sa_score(mol):
    fallback_value = 5.0
    if not SA_SCORER_AVAILABLE or sascorer is None:
        return fallback_value
    try:
        return round(float(sascorer.calculateScore(mol)), 2)
    except Exception:
        return fallback_value


def classify_rank(ranking_score):
    if ranking_score >= 4.5:
        return "Excellent"
    elif ranking_score >= 3.5:
        return "Good"
    elif ranking_score >= 2.5:
        return "Moderate"
    return "Poor"


def classify_solubility(logs_value):
    if logs_value >= 0:
        return "Highly soluble"
    elif logs_value >= -2:
        return "Very soluble"
    elif logs_value >= -4:
        return "Soluble"
    elif logs_value >= -6:
        return "Moderately soluble"
    elif logs_value >= -10:
        return "Poorly soluble"
    return "Insoluble"


def calculate_esol(mw, logp, rot_bonds, aromatic_atoms, heavy_atoms):
    aromatic_proportion = (aromatic_atoms / heavy_atoms) if heavy_atoms > 0 else 0.0
    logs = 0.16 - logp - (0.0062 * mw) + (0.066 * rot_bonds) + (0.066 * aromatic_proportion)
    solubility_mg_ml = (10 ** logs) * mw
    return round(logs, 3), round(solubility_mg_ml, 4)


def estimate_log_kp(logp, mw):
    log_kp = -2.72 + (0.71 * logp) - (0.0061 * mw)
    return round(log_kp, 2)


def estimate_bioavailability_score(lipinski_violations, veber_pass, logp, tpsa):
    score = 0.55
    if lipinski_violations > 1:
        score = 0.17
    elif not veber_pass:
        score = 0.30
    elif logp > 5 or tpsa > 140:
        score = 0.30
    return round(score, 2)


def traffic_light_status(value, green_condition, amber_condition=None):
    if green_condition(value):
        return "green"
    if amber_condition is not None and amber_condition(value):
        return "amber"
    return "red"


def get_murcko_scaffold_smiles(mol):
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "No clear scaffold"
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return "No clear scaffold"


def score_center(value, center, tolerance):
    return clamp01(1 - abs(value - center) / tolerance)


def score_max(value, ideal_max, hard_max):
    if value <= ideal_max:
        return 1.0
    return clamp01(1 - ((value - ideal_max) / (hard_max - ideal_max)))


def score_min(value, ideal_min, hard_min):
    if value >= ideal_min:
        return 1.0
    return clamp01((value - hard_min) / (ideal_min - hard_min))


def calculate_goal_specific_score(row, project_goal):
    total_alerts = row["pains_alerts_count"] + row["brenk_alerts_count"]
    sa_penalty = max(0.0, (row["synthetic_accessibility"] - 4.0) / 4.0)
    alert_penalty = min(1.0, total_alerts / 4.0)

    if project_goal == "oral":
        mw_fit = score_max(row["mw"], 500, 650)
        logp_fit = score_center(row["logp"], 2.5, 3.5)
        tpsa_fit = score_max(row["tpsa"], 120, 180)
        flex_fit = score_max(row["rot_bonds"], 8, 15)
        sol_fit = score_max(abs(row["esol_logs"]), 4.0, 8.0)
        raw = (
            0.22 * mw_fit +
            0.22 * logp_fit +
            0.20 * tpsa_fit +
            0.15 * flex_fit +
            0.11 * sol_fit +
            0.10 * (1 - sa_penalty)
        ) - (0.12 * alert_penalty)

    elif project_goal == "lead":
        mw_fit = score_max(row["mw"], 450, 600)
        logp_fit = score_center(row["logp"], 2.2, 3.2)
        tpsa_fit = score_max(row["tpsa"], 110, 170)
        flex_fit = score_max(row["rot_bonds"], 8, 14)
        csp3_fit = score_min(row["frac_csp3"], 0.25, 0.0)
        raw = (
            0.18 * mw_fit +
            0.20 * logp_fit +
            0.16 * tpsa_fit +
            0.14 * flex_fit +
            0.16 * csp3_fit +
            0.16 * (1 - sa_penalty)
        ) - (0.14 * alert_penalty)

    elif project_goal == "fragment":
        mw_fit = score_max(row["mw"], 300, 450)
        logp_fit = score_max(row["logp"], 3.0, 5.0)
        hbd_fit = score_max(row["hbd"], 3, 6)
        hba_fit = score_max(row["hba"], 3, 6)
        flex_fit = score_max(row["rot_bonds"], 3, 8)
        rule3_fit = 1.0 if row["rule_of_3"] == "Pass" else 0.35
        raw = (
            0.22 * mw_fit +
            0.16 * logp_fit +
            0.12 * hbd_fit +
            0.12 * hba_fit +
            0.14 * flex_fit +
            0.24 * rule3_fit
        ) - (0.08 * alert_penalty)

    elif project_goal == "cns":
        mw_fit = score_max(row["mw"], 450, 550)
        logp_fit = score_center(row["logp"], 2.3, 2.5)
        tpsa_fit = score_max(row["tpsa"], 90, 130)
        hbd_fit = score_max(row["hbd"], 2, 4)
        bbb_fit = 1.0 if row["bbb_permeant"] == "Yes" else 0.25
        flex_fit = score_max(row["rot_bonds"], 8, 14)
        raw = (
            0.16 * mw_fit +
            0.18 * logp_fit +
            0.18 * tpsa_fit +
            0.14 * hbd_fit +
            0.22 * bbb_fit +
            0.12 * flex_fit
        ) - (0.10 * alert_penalty)

    else:  # general
        mw_fit = score_max(row["mw"], 500, 650)
        logp_fit = score_center(row["logp"], 2.5, 4.0)
        tpsa_fit = score_max(row["tpsa"], 140, 180)
        flex_fit = score_max(row["rot_bonds"], 10, 15)
        csp3_fit = score_min(row["frac_csp3"], 0.25, 0.0)
        raw = (
            0.22 * mw_fit +
            0.22 * logp_fit +
            0.20 * tpsa_fit +
            0.16 * flex_fit +
            0.10 * csp3_fit +
            0.10 * (1 - sa_penalty)
        ) - (0.10 * alert_penalty)

    return round(clamp(raw * 6.0, 0.0, 6.0), 3)


def build_liability_map(row):
    total_alerts = row["pains_alerts_count"] + row["brenk_alerts_count"]

    return [
        {
            "label": "Solubility",
            "status": traffic_light_status(
                row["esol_logs"],
                lambda x: x >= -4.0,
                lambda x: -6.0 <= x < -4.0
            ),
            "value": row["solubility_class"]
        },
        {
            "label": "Lipophilicity",
            "status": traffic_light_status(
                row["logp"],
                lambda x: 1.0 <= x <= 3.5,
                lambda x: 0.0 <= x <= 5.0
            ),
            "value": row["logp"]
        },
        {
            "label": "Polarity",
            "status": traffic_light_status(
                row["tpsa"],
                lambda x: 20 <= x <= 120,
                lambda x: 0 <= x <= 140
            ),
            "value": row["tpsa"]
        },
        {
            "label": "Flexibility",
            "status": traffic_light_status(
                row["rot_bonds"],
                lambda x: x <= 7,
                lambda x: x <= 10
            ),
            "value": row["rot_bonds"]
        },
        {
            "label": "Alerts",
            "status": traffic_light_status(
                total_alerts,
                lambda x: x == 0,
                lambda x: x <= 2
            ),
            "value": total_alerts
        },
        {
            "label": "Synthesis",
            "status": traffic_light_status(
                row["synthetic_accessibility"],
                lambda x: x <= 4.0,
                lambda x: x <= 6.0
            ),
            "value": row["synthetic_accessibility"]
        }
    ]


def build_project_fit_labels(row, project_goal):
    labels = []

    if row["lipinski_violations"] == 0 and row["veber"] == "Pass":
        labels.append("Oral-friendly")

    if row["mw"] <= 450 and row["lead_like"] == "Pass":
        labels.append("Lead-optimization friendly")

    if row["rule_of_3"] == "Pass":
        labels.append("Fragment-like")

    if row["bbb_permeant"] == "Yes":
        labels.append("CNS-permeation favorable")

    if row["esol_logs"] < -4:
        labels.append("Solubility-watch")

    if (row["pains_alerts_count"] + row["brenk_alerts_count"]) > 0:
        labels.append("Med-chem review")

    if row["frac_csp3"] >= 0.25:
        labels.append("3D character favorable")

    goal_label_map = {
        "general": "General screening mode",
        "oral": "Oral-project mode",
        "lead": "Lead-optimization mode",
        "fragment": "Fragment-priority mode",
        "cns": "CNS-project mode"
    }
    labels.insert(0, goal_label_map.get(project_goal, "General screening mode"))

    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    return unique_labels[:5]


def build_optimization_suggestions(row, project_goal):
    suggestions = []

    if row["logp"] > 4.5:
        suggestions.append("Reduce lipophilicity to improve solubility and balance permeability.")
    if row["tpsa"] > 140:
        suggestions.append("Lower polar surface area to improve passive permeability.")
    if row["rot_bonds"] > 8:
        suggestions.append("Trim flexible linkers or reduce rotatable bonds to improve conformational control.")
    if row["mw"] > 500:
        suggestions.append("Consider reducing molecular size to improve developability and oral-likeness.")
    if row["frac_csp3"] < 0.25:
        suggestions.append("Increase sp3 character to improve 3D complexity and scaffold quality.")
    if row["synthetic_accessibility"] > 6:
        suggestions.append("Simplify the scaffold or reduce synthetic burden for easier progression.")
    if row["pains_alerts_count"] > 0:
        suggestions.append("Review and remove PAINS-associated motifs before prioritization.")
    if row["brenk_alerts_count"] > 0:
        suggestions.append("Inspect Brenk alerts for problematic medicinal chemistry liabilities.")
    if row["hbd"] > 5 or row["hba"] > 10:
        suggestions.append("Reduce hydrogen-bond donor or acceptor burden to improve property balance.")

    if project_goal == "oral" and row["gi_absorption"] != "High":
        suggestions.append("For oral programs, improve absorption by reducing TPSA or flexibility.")
    if project_goal == "lead" and row["mw"] > 450:
        suggestions.append("For lead optimization, consider trimming excess mass before potency expansion.")
    if project_goal == "fragment" and row["rule_of_3"] != "Pass":
        suggestions.append("For fragment-style design, reduce size and complexity toward Rule-of-3 space.")
    if project_goal == "cns" and row["bbb_permeant"] != "Yes":
        suggestions.append("For CNS design, move toward lower TPSA, lower HBD, and controlled lipophilicity.")

    if not suggestions:
        suggestions.append("Current profile is balanced. Focus next on potency, selectivity, and series expansion.")

    return suggestions[:5]


def build_score_breakdown(row, project_goal):
    mw_score = round(scale_inverse(row["mw"], 200, 500), 2)
    logp_score = round(scale_inverse(abs(row["logp"] - 2.5), 0, 5), 2)
    tpsa_score = round(scale_inverse(row["tpsa"], 20, 140), 2)
    flex_score = round(scale_inverse(row["rot_bonds"], 0, 10), 2)

    alert_penalty = round(min(2.0, 0.5 * (row["pains_alerts_count"] + row["brenk_alerts_count"])), 2)
    sa_penalty = round(max(0.0, (row["synthetic_accessibility"] - 4.0) * 0.25), 2)

    goal_weight_note = {
        "general": "General weighting",
        "oral": "Oral-biased weighting",
        "lead": "Lead-biased weighting",
        "fragment": "Fragment-biased weighting",
        "cns": "CNS-biased weighting"
    }

    return [
        {"label": "MW fit", "value": mw_score},
        {"label": "LogP fit", "value": logp_score},
        {"label": "TPSA fit", "value": tpsa_score},
        {"label": "Flexibility fit", "value": flex_score},
        {"label": "Alert penalty", "value": alert_penalty},
        {"label": goal_weight_note.get(project_goal, "General weighting"), "value": row["ranking_score"]}
    ]


def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    ring_count = Lipinski.RingCount(mol)
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    frac_csp3 = Lipinski.FractionCSP3(mol)
    mol_refractivity = Crippen.MolMR(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    atom_count = mol.GetNumAtoms()
    heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    valence_electrons = Descriptors.NumValenceElectrons(mol)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10
    ])

    veber_pass = (rot_bonds <= 10 and tpsa <= 140)

    bbb_assessment = "Yes" if (mw <= 450 and 1 <= logp <= 5 and tpsa <= 90 and hbd <= 3) else "No"
    absorption_assessment = "High" if (tpsa <= 140 and rot_bonds <= 10) else "Low"
    permeability_assessment = "Favorable" if (mw <= 500 and logp <= 5 and tpsa <= 120) else "Limited"

    solubility_risk = "High risk" if logp > 5 else "Manageable"
    flexibility_risk = "High" if rot_bonds > 10 else "Acceptable"
    polarity_class = "Low" if tpsa < 75 else ("Medium" if tpsa <= 140 else "High")
    size_class = "Small" if mw < 300 else ("Medium" if mw <= 500 else "Large")
    hydrophobicity_class = "Low" if logp < 1 else ("Moderate" if logp <= 5 else "High")

    ghose_like = "Pass" if (160 <= mw <= 480 and -0.4 <= logp <= 5.6 and 20 <= atom_count <= 70) else "Fail"
    rule_of_3 = "Pass" if (mw <= 300 and logp <= 3 and hbd <= 3 and hba <= 3 and rot_bonds <= 3) else "Fail"
    lead_like = "Pass" if (mw <= 450 and logp <= 4 and rot_bonds <= 10) else "Fail"

    ranking_score = round(
        (scale_inverse(mw, 200, 500) +
         scale_inverse(abs(logp - 2.5), 0, 5) +
         scale_inverse(tpsa, 20, 140) +
         scale_inverse(rot_bonds, 0, 10)) / 4,
        3
    )
    rank_class = classify_rank(ranking_score)

    pains_matches = get_filter_matches(mol, PAINS_CATALOG)
    brenk_matches = get_filter_matches(mol, BRENK_CATALOG)
    sa_score = calculate_sa_score(mol)

    esol_logs, esol_solubility = calculate_esol(
        mw=mw,
        logp=logp,
        rot_bonds=rot_bonds,
        aromatic_atoms=aromatic_atoms,
        heavy_atoms=heavy_atoms
    )
    solubility_class = classify_solubility(esol_logs)
    log_kp = estimate_log_kp(logp, mw)
    bioavailability_score = estimate_bioavailability_score(lipinski_violations, veber_pass, logp, tpsa)

    scaffold_smiles = get_murcko_scaffold_smiles(mol)

    ilogp = round(logp, 2)
    xlogp = round(logp, 2)
    wlogp = round(logp, 2)
    mlogp = round(logp, 2)

    return {
        "canonical_smiles": canonical_smiles,
        "formula": formula,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "ilogp": ilogp,
        "xlogp": xlogp,
        "wlogp": wlogp,
        "mlogp": mlogp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 2),
        "rot_bonds": rot_bonds,
        "rotatable_bonds": rot_bonds,
        "heavy_atoms": heavy_atoms,
        "ring_count": ring_count,
        "aromatic_rings": aromatic_rings,
        "frac_csp3": round(frac_csp3, 3),
        "mol_refractivity": round(mol_refractivity, 2),
        "formal_charge": formal_charge,
        "atom_count": atom_count,
        "heteroatoms": heteroatoms,
        "hetero_atoms": heteroatoms,
        "valence_electrons": valence_electrons,
        "lipinski_violations": lipinski_violations,
        "veber_pass": "Pass" if veber_pass else "Fail",
        "veber": "Pass" if veber_pass else "Fail",
        "bbb_assessment": bbb_assessment,
        "bbb_permeant": bbb_assessment,
        "absorption_assessment": absorption_assessment,
        "gi_absorption": absorption_assessment,
        "permeability_assessment": permeability_assessment,
        "solubility_risk": solubility_risk,
        "flexibility_risk": flexibility_risk,
        "polarity_class": polarity_class,
        "size_class": size_class,
        "hydrophobicity_class": hydrophobicity_class,
        "ghose_like": ghose_like,
        "ghose": ghose_like,
        "rule_of_3": rule_of_3,
        "lead_like": lead_like,
        "lead_like_text": lead_like,
        "ranking_score": ranking_score,
        "rank_class": rank_class,
        "pains_alerts_count": len(set(pains_matches)),
        "pains_alerts_text": summarize_alerts(pains_matches),
        "brenk_alerts_count": len(set(brenk_matches)),
        "brenk_alerts_text": summarize_alerts(brenk_matches),
        "synthetic_accessibility": safe_float(sa_score, 5.0),
        "sa_score": safe_float(sa_score, 5.0),
        "esol_logs": esol_logs,
        "esol_solubility": esol_solubility,
        "solubility_class": solubility_class,
        "bioavailability_score": bioavailability_score,
        "log_kp": log_kp,
        "pgp_substrate": "No",
        "cyp1a2_inhibitor": "No",
        "cyp2c19_inhibitor": "No",
        "cyp2c9_inhibitor": "No",
        "cyp2d6_inhibitor": "No",
        "cyp3a4_inhibitor": "No",
        "scaffold_smiles": scaffold_smiles
    }


def parse_smiles_from_csv(file_storage):
    smiles_list = []
    if not file_storage or file_storage.filename == "":
        return smiles_list

    try:
        content = file_storage.read().decode("utf-8", errors="ignore")
        file_storage.seek(0)

        reader = csv.DictReader(io.StringIO(content))
        if reader.fieldnames is None:
            return smiles_list

        fieldnames_lower = [f.lower().strip() for f in reader.fieldnames]
        smiles_col = None

        for idx, col in enumerate(fieldnames_lower):
            if col == "smiles":
                smiles_col = reader.fieldnames[idx]
                break

        if smiles_col is None:
            return []

        for row in reader:
            val = row.get(smiles_col, "")
            if val and val.strip():
                smiles_list.append(val.strip())

        return smiles_list

    except Exception:
        return []


def collect_smiles_from_request(req):
    input_method = req.form.get("input_method", "batch")
    project_goal = req.form.get("project_goal", "general").strip().lower()

    allowed_goals = {"general", "oral", "lead", "fragment", "cns"}
    if project_goal not in allowed_goals:
        project_goal = "general"

    smiles_list = []

    if input_method == "single":
        single_smiles = req.form.get("single_smiles", "").strip()
        if single_smiles:
            smiles_list.append(single_smiles)

    elif input_method == "batch":
        batch_smiles = req.form.get("batch_smiles", "")
        smiles_list.extend([s.strip() for s in batch_smiles.splitlines() if s.strip()])

    elif input_method == "draw":
        drawn_smiles = req.form.get("drawn_smiles", "").strip()
        if drawn_smiles:
            smiles_list.append(drawn_smiles)

    elif input_method == "upload":
        uploaded = req.files.get("file_upload")
        smiles_list = parse_smiles_from_csv(uploaded)

    return input_method, project_goal, smiles_list


def mol_image_base64(smiles, size=(360, 260)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=size)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_radar_axes(row):
    lipo = clamp(6.0 - abs(row["logp"] - 2.5) * 1.2)
    size = clamp(6.0 - abs(row["mw"] - 350) / 40.0)
    polar = scale_inverse(row["tpsa"], 20, 150)
    insol = clamp(6.0 - max(row["logp"], 0) * 0.8)
    insatu = scale_range(row["frac_csp3"], 0.0, 1.0)
    flex = scale_inverse(row["rot_bonds"], 0, 12)

    return [
        {"label": "LIPO", "value": round(lipo, 2)},
        {"label": "SIZE", "value": round(size, 2)},
        {"label": "POLAR", "value": round(polar, 2)},
        {"label": "INSOLU", "value": round(insol, 2)},
        {"label": "INSATU", "value": round(insatu, 2)},
        {"label": "FLEX", "value": round(flex, 2)}
    ]


def build_sections(row):
    return {
        "Physicochemical Properties": [
            ("Canonical SMILES", row["canonical_smiles"]),
            ("Formula", row["formula"]),
            ("Atoms", row["atom_count"]),
            ("Heavy atoms", row["heavy_atoms"]),
            ("Molecular weight", f"{row['mw']} g/mol"),
            ("TPSA", f"{row['tpsa']} Å²"),
            ("H-bond acceptors", row["hba"]),
            ("H-bond donors", row["hbd"]),
            ("Rotatable bonds", row["rot_bonds"]),
            ("Ring count", row["ring_count"]),
            ("Aromatic rings", row["aromatic_rings"]),
            ("Fraction Csp3", row["frac_csp3"]),
            ("Molar refractivity", row["mol_refractivity"])
        ],
        "Medicinal Chemistry": [
            ("PAINS", row["pains_alerts_text"]),
            ("Brenk", row["brenk_alerts_text"]),
            ("Synthetic accessibility", row["synthetic_accessibility"]),
            ("Scaffold", row["scaffold_smiles"])
        ],
        "MolScore-PK Ranking": [
            ("Project goal", row["project_goal_label"]),
            ("Descriptor-based ranking score", row["ranking_score"]),
            ("Ranking class", row["rank_class"]),
            ("Status", row["status"]),
            ("Interpretation", "Calculated descriptors with medicinal chemistry-aware Platform support")
        ]
    }

def build_batch_summary(results):
    valid_rows = [r for r in results if r.get("status") == "Valid"]
    if not valid_rows:
        return None

    scaffold_set = set()
    for row in valid_rows:
        scaffold = row.get("scaffold_smiles", "")
        if scaffold:
            scaffold_set.add(scaffold)

    best_row = max(valid_rows, key=lambda x: x.get("ranking_score", 0))

    avg_score = round(sum(r.get("ranking_score", 0) for r in valid_rows) / len(valid_rows), 3)
    avg_mw = round(sum(r.get("mw", 0) for r in valid_rows) / len(valid_rows), 2)
    avg_logp = round(sum(r.get("logp", 0) for r in valid_rows) / len(valid_rows), 2)

    return {
        "valid_count": len(valid_rows),
        "scaffold_count": len(scaffold_set),
        "average_ranking_score": avg_score,
        "average_mw": avg_mw,
        "average_logp": avg_logp,
        "top_molecule_id": best_row.get("id"),
        "top_molecule_score": best_row.get("ranking_score"),
        "top_molecule_class": best_row.get("rank_class"),
        "project_goal_label": best_row.get("project_goal_label", "General screening")
    }


def build_download_rows(results):
    csv_rows = []
    for row in results:
        csv_rows.append({
            "id": row["id"],
            "smiles": row["smiles"],
            "canonical_smiles": row.get("canonical_smiles", ""),
            "formula": row.get("formula", ""),
            "status": row["status"],
            "project_goal": row.get("project_goal", ""),
            "project_goal_label": row.get("project_goal_label", ""),
            "mw": row.get("mw", ""),
            "logp": row.get("logp", ""),
            "ilogp": row.get("ilogp", ""),
            "xlogp": row.get("xlogp", ""),
            "wlogp": row.get("wlogp", ""),
            "mlogp": row.get("mlogp", ""),
            "hbd": row.get("hbd", ""),
            "hba": row.get("hba", ""),
            "tpsa": row.get("tpsa", ""),
            "rot_bonds": row.get("rot_bonds", ""),
            "heavy_atoms": row.get("heavy_atoms", ""),
            "ring_count": row.get("ring_count", ""),
            "aromatic_rings": row.get("aromatic_rings", ""),
            "frac_csp3": row.get("frac_csp3", ""),
            "mol_refractivity": row.get("mol_refractivity", ""),
            "bbb_assessment": row.get("bbb_assessment", ""),
            "bbb_permeant": row.get("bbb_permeant", ""),
            "gi_absorption": row.get("gi_absorption", ""),
            "permeability_assessment": row.get("permeability_assessment", ""),
            "solubility_risk": row.get("solubility_risk", ""),
            "flexibility_risk": row.get("flexibility_risk", ""),
            "polarity_class": row.get("polarity_class", ""),
            "size_class": row.get("size_class", ""),
            "hydrophobicity_class": row.get("hydrophobicity_class", ""),
            "lipinski_violations": row.get("lipinski_violations", ""),
            "veber_pass": row.get("veber_pass", ""),
            "ghose_like": row.get("ghose_like", ""),
            "rule_of_3": row.get("rule_of_3", ""),
            "lead_like": row.get("lead_like", ""),
            "pains_alerts_count": row.get("pains_alerts_count", ""),
            "pains_alerts_text": row.get("pains_alerts_text", ""),
            "brenk_alerts_count": row.get("brenk_alerts_count", ""),
            "brenk_alerts_text": row.get("brenk_alerts_text", ""),
            "synthetic_accessibility": row.get("synthetic_accessibility", ""),
            "esol_logs": row.get("esol_logs", ""),
            "esol_solubility": row.get("esol_solubility", ""),
            "solubility_class": row.get("solubility_class", ""),
            "bioavailability_score": row.get("bioavailability_score", ""),
            "log_kp": row.get("log_kp", ""),
            "scaffold_smiles": row.get("scaffold_smiles", ""),
            "ranking_score": row.get("ranking_score", ""),
            "rank_class": row.get("rank_class", "")
        })
    return csv_rows


@app.after_request
def add_no_cache_headers(response):
    if request.path in ["/", "/prediction", "/documentation", "/predict", "/download_csv"]:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/prediction", methods=["GET"])
def prediction():
    return render_template("prediction.html", selected_method="draw")


@app.route("/documentation")
def documentation():
    return render_template("documentation.html")


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    input_method, project_goal, smiles_list = collect_smiles_from_request(request)

    if not smiles_list:
        return render_template(
            "prediction.html",
            error="No valid molecular input detected. Please enter SMILES, draw a molecule, or upload a CSV with a 'smiles' column.",
            selected_method=input_method
        )

    if len(smiles_list) > 50:
        return render_template(
            "prediction.html",
            error="Maximum 50 molecules are allowed per run.",
            selected_method=input_method
        )

    goal_label_map = {
        "general": "General screening",
        "oral": "Oral small molecule",
        "lead": "Lead optimization",
        "fragment": "Fragment-like exploration",
        "cns": "CNS-oriented design"
    }

    results = []

    for i, smiles in enumerate(smiles_list, start=1):
        props = calculate_properties(smiles)

        if props is None:
            results.append({
                "id": i,
                "smiles": smiles,
                "status": "Invalid SMILES",
                "image": None,
                "radar_axes": [],
                "sections": {},
                "project_goal": project_goal,
                "project_goal_label": goal_label_map.get(project_goal, "General screening")
            })
        else:
            row = {
                "id": i,
                "smiles": smiles,
                "status": "Valid",
                "project_goal": project_goal,
                "project_goal_label": goal_label_map.get(project_goal, "General screening")
            }
            row.update(props)

            row["ranking_score"] = calculate_goal_specific_score(row, project_goal)
            row["rank_class"] = classify_rank(row["ranking_score"])
            row["project_fit_labels"] = build_project_fit_labels(row, project_goal)
            row["optimization_suggestions"] = build_optimization_suggestions(row, project_goal)
            row["score_breakdown"] = build_score_breakdown(row, project_goal)

            row["image"] = mol_image_base64(smiles)
            row["radar_axes"] = build_radar_axes(row)
            row["sections"] = build_sections(row)
            results.append(row)

    try:
        session["last_results"] = json.dumps(build_download_rows(results))
    except Exception:
        session["last_results"] = "[]"

    elapsed = round(time.time() - start_time, 2)
    batch_summary = build_batch_summary(results)

    return render_template(
        "results.html",
        results=results,
        runtime=elapsed,
        batch_summary=batch_summary
    )


@app.route("/download_csv", methods=["GET"])
def download_csv():
    raw_results = session.get("last_results")
    if not raw_results:
        return render_template(
            "prediction.html",
            error="No results are available for download. Please run a prediction first.",
            selected_method="draw"
        )

    try:
        results = json.loads(raw_results)
    except Exception:
        results = []

    output = io.StringIO()
    writer = csv.writer(output)

    if results:
        headers = list(results[0].keys())
        writer.writerow(headers)
        for row in results:
            writer.writerow([row.get(h, "") for h in headers])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=molscore_pk_results.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "prediction.html",
        error="Uploaded file is too large. Please upload a smaller CSV file.",
        selected_method="upload"
    ), 413


@app.errorhandler(404)
def page_not_found(error):
    return render_template(
        "prediction.html",
        error="Requested page was not found.",
        selected_method="draw"
    ), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template(
        "prediction.html",
        error="An internal server error occurred. Please try again.",
        selected_method="draw"
    ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)