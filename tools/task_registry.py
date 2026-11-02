"""
Daftar semua template (task) di repo. Dipakai bersama oleh tools/new_task.py, tools/run_task.py dan
tools/smoke_test_templates.py supaya nama task, lokasi file, dan contoh config hanya ditulis di satu tempat.

Setiap template tersedia dalam dua format dengan nama sama: <nama>.ipynb (sumber) dan <nama>.py (dibuat
otomatis oleh tools/nb_formats.py).
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_COMMON_TABLE = {"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv",
                 "SAMPLE_SUB_PATH": "./data/sample_submission.csv"}

TASKS = {
    # ---------------------------------------------------------------- tabular & bisnis
    "churn": {
        "path": "01_Churn_Insurance/churn_insurance_master.ipynb",
        "title": "Churn / retensi / asuransi + artikel, dashboard, ZIP otomatis",
        "special": "Pakai tools/new_case.py (alur lengkap hari-H) — bukan new_task.py.",
        "config": {},
    },
    "tabular": {
        "path": "02_Tabular/tabular_master.ipynb",
        "title": "Tabular: klasifikasi biner/multikelas, regresi",
        "data": "data/train.csv, data/test.csv, data/sample_submission.csv",
        "config": {**_COMMON_TABLE, "ID_COL": "id", "TARGET_COL": "target", "TASK": "auto", "METRIC": "auto"},
    },
    "tabular-multilabel": {
        "path": "02_Tabular/tabular_multilabel.ipynb",
        "title": "Tabular multilabel: beberapa target 0/1 sekaligus (atau satu kolom berisi daftar label)",
        "data": "data/train.csv (fitur + kolom label), data/test.csv, data/sample_submission.csv",
        "config": {**_COMMON_TABLE, "ID_COL": "id", "TARGET_COL": None},
    },
    "stats": {
        "path": "09_Statistical_Analysis/statistical_analysis_report.ipynb",
        "title": "Analisis statistik & laporan insight (uji hipotesis, A/B test, regresi + CI) tanpa lomba prediksi",
        "data": "data/data.csv (satu tabel; boleh xlsx/parquet)",
        "config": {"DATA_PATH": "./data/data.csv", "OUTCOME_COL": None, "GROUP_COL": None},
    },
    "anomaly-tabular": {
        "path": "07_Anomaly_Detection/anomaly_tabular.ipynb",
        "title": "Anomaly / fraud / outlier detection pada data tabular (tanpa label atau sedikit label)",
        "data": "data/train.csv (+ data/test.csv opsional; kolom label opsional)",
        "config": {"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "ID_COL": "id", "LABEL_COL": None},
    },
    "time-series": {
        "path": "05_Time_Series/time_series_forecasting.ipynb",
        "title": "Forecasting deret waktu (single series & panel)",
        "data": "data/train.csv (tanggal, [id seri], target), data/test.csv opsional",
        "config": {**_COMMON_TABLE, "DATE_COL": None, "TARGET_COL": "sales", "SERIES_ID_COLS": [], "HORIZON": None},
    },
    "clustering": {
        "path": "06_Clustering_Segmentation/clustering_segmentation.ipynb",
        "title": "Clustering / segmentasi customer tanpa label (+ RFM)",
        "data": "data/train.csv",
        "config": {"DATA_PATH": "./data/train.csv", "ID_COL": "id"},
    },
    # ---------------------------------------------------------------- teks
    "nlp-classification": {
        "path": "03_NLP/nlp_classification.ipynb",
        "title": "Teks → kelas (biner/multikelas/multilabel): TF-IDF, embedding, transformer",
        "data": "data/train.csv (id, text, label), data/test.csv",
        "config": {**_COMMON_TABLE, "ID_COL": "id", "TEXT_COL": "text", "TARGET_COL": "label"},
    },
    "nlp-regression": {
        "path": "03_NLP/nlp_regression.ipynb",
        "title": "Teks → angka (skor, harga, rating)",
        "data": "data/train.csv (id, text, target), data/test.csv",
        "config": {**_COMMON_TABLE, "ID_COL": "id", "TEXT_COL": "text", "TARGET_COL": "target"},
    },
    "ner": {
        "path": "03_NLP/nlp_token_classification.ipynb",
        "title": "NER / slot filling / token classification (BIO)",
        "data": "data/train.conll|.txt|.jsonl|.csv, data/test.*",
        "config": {"TRAIN_PATH": "./data/train.conll", "TEST_PATH": "./data/test.conll"},
    },
    "llm": {
        "path": "03_NLP/nlp_llm_zero_shot.ipynb",
        "title": "LLM lokal: zero/few-shot classification & ekstraksi (+ LoRA opsional)",
        "data": "data/test.csv (id, text), data/train.csv opsional (contoh few-shot & evaluasi)",
        "config": {"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "TEXT_COL": "text",
                   "TARGET_COL": "label", "LABELS": None},
    },
    # ---------------------------------------------------------------- gambar
    "cv-classification": {
        "path": "04_Computer_Vision/cv_image_classification.ipynb",
        "title": "Gambar → kelas",
        "data": "data/train.csv + data/train/ (gambar) atau data/train/<kelas>/*.jpg, data/test/",
        "config": {"TRAIN_CSV": "./data/train.csv", "TEST_CSV": "./data/test.csv",
                   "TRAIN_IMAGE_DIR": "./data/train", "TEST_IMAGE_DIR": "./data/test"},
    },
    "segmentation": {
        "path": "04_Computer_Vision/cv_segmentation.ipynb",
        "title": "Segmentasi gambar (semantic, biner/multikelas): mask PNG atau RLE",
        "data": "data/train/images/, data/train/masks/ (atau train.csv berisi RLE), data/test/images/",
        "config": {"TRAIN_IMAGE_DIR": "./data/train/images", "TRAIN_MASK_DIR": "./data/train/masks",
                   "TEST_IMAGE_DIR": "./data/test/images", "SAMPLE_SUB_PATH": "./data/sample_submission.csv"},
    },
    "detection": {
        "path": "04_Computer_Vision/cv_object_detection.ipynb",
        "title": "Object detection & counting (YOLO): bounding box atau jumlah objek",
        "data": "data/train/images + labels (YOLO/COCO/VOC/CSV), data/test/images",
        "config": {"TRAIN_IMAGE_DIR": "./data/train/images", "ANNOTATION_PATH": None,
                   "TEST_IMAGE_DIR": "./data/test/images", "TASK_MODE": "detect"},
    },
    "ocr": {
        "path": "04_Computer_Vision/cv_ocr_text_recognition.ipynb",
        "title": "OCR / text recognition: digit, plat, aksara, teks pendek dalam gambar (CRNN-CTC)",
        "data": "data/train.csv (image, text) + data/train/, data/test/",
        "config": {"TRAIN_CSV": "./data/train.csv", "TRAIN_IMAGE_DIR": "./data/train",
                   "TEST_CSV": "./data/test.csv", "TEST_IMAGE_DIR": "./data/test"},
    },
    "anomaly-image": {
        "path": "07_Anomaly_Detection/anomaly_image.ipynb",
        "title": "Anomaly detection gambar (cacat produk/visual inspection, hanya contoh normal)",
        "data": "data/train/good/ (gambar normal), data/test/ (gambar campur)",
        "config": {"TRAIN_IMAGE_DIR": "./data/train/good", "TEST_IMAGE_DIR": "./data/test"},
    },
    # ---------------------------------------------------------------- audio
    "audio": {
        "path": "08_Audio/audio_classification.ipynb",
        "title": "Audio → kelas (suara hewan, ucapan, mesin) + open-set 'Unknown'",
        "data": "data/train.csv (file, label) + data/train/ (wav/flac/mp3) atau data/train/<kelas>/, data/test/",
        "config": {"TRAIN_CSV": "./data/train.csv", "TRAIN_AUDIO_DIR": "./data/train",
                   "TEST_AUDIO_DIR": "./data/test", "SAMPLE_SUB_PATH": "./data/sample_submission.csv"},
    },
}


def template_path(key, fmt="ipynb"):
    p = os.path.join(REPO, TASKS[key]["path"])
    return os.path.splitext(p)[0] + (".py" if fmt == "py" else ".ipynb")


def template_name(key):
    return os.path.splitext(os.path.basename(TASKS[key]["path"]))[0]


def available(key):
    return os.path.exists(template_path(key, "ipynb"))


def describe():
    lines = []
    for k, t in TASKS.items():
        flag = "" if available(k) else "  (belum ada)"
        lines.append(f"  {k:20s} {t['title']}{flag}")
    return "\n".join(lines)
