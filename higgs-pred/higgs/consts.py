from pathlib import Path


COLNAMES = [
    "label",
    "lepton_pt",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet_1_pt",
    "jet_1_eta",
    "jet_1_phi",
    "jet_1_b-tag",
    "jet_2_pt",
    "jet_2_eta",
    "jet_2_phi",
    "jet_2_b-tag",
    "jet_3_pt",
    "jet_3_eta",
    "jet_3_phi",
    "jet_3_b-tag",
    "jet_4_pt",
    "jet_4_eta",
    "jet_4_phi",
    "jet_4_b-tag",
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
]

DATAROOT = Path.home() / "mldata"
RUNROOT = Path.home() / "mlruns"
PROJECT = "higgs"
HIGGS_CSV = "HIGGS.csv"
HIGGS_TRAIN_CSV = "HIGGS_TRAIN.csv"
HIGGS_TEST_CSV = "HIGGS_TEST.csv"
SAMPLE_HIGGS_CSV = "SAMPLE_HIGGS.csv"
