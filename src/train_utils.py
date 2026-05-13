from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test_x.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

SUBMISSION_DIR = ROOT_DIR / "submissions"
MODEL_DIR = ROOT_DIR / "model"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "bilissel_performans_skoru"
ID_COL = "id"

RANDOM_STATE = 42
N_SPLITS = 5