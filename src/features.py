import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    YZTA Datathon verisi için domain-based feature engineering fonksiyonu.
    Hem train hem test için aynı şekilde kullanılmalıdır.
    """

    df = df.copy()

    # Uyku kalitesi feature'ları
    if {"rem_yuzdesi", "derin_uyku_yuzdesi"}.issubset(df.columns):
        df["toplam_kaliteli_uyku_yuzdesi"] = (
            df["rem_yuzdesi"] + df["derin_uyku_yuzdesi"]
        )

        df["rem_derin_uyku_carpim"] = (
            df["rem_yuzdesi"] * df["derin_uyku_yuzdesi"]
        )

    # Uyku bölünmesi / uyku verimsizliği
    if {"gecelik_uyanma_sayisi", "uykuya_dalma_suresi_dk"}.issubset(df.columns):
        df["uyku_bolunme_yuku"] = (
            df["gecelik_uyanma_sayisi"] * df["uykuya_dalma_suresi_dk"]
        )

        df["uyku_verimsizlik_skoru"] = (
            df["uykuya_dalma_suresi_dk"] + 10 * df["gecelik_uyanma_sayisi"]
        )

    # Stres ve çalışma yükü
    if {"stres_skoru", "gunluk_calisma_saati"}.issubset(df.columns):
        df["stres_calisma_yuku"] = (
            df["stres_skoru"] * df["gunluk_calisma_saati"]
        )

    # Ekran + kafein yükü
    if {"uyku_oncesi_ekran_suresi_dk", "uyku_oncesi_kafein_mg"}.issubset(df.columns):
        df["ekran_kafein_yuku"] = (
            df["uyku_oncesi_ekran_suresi_dk"] + df["uyku_oncesi_kafein_mg"]
        )

    # Adım sayısını daha okunabilir ölçeğe çekiyoruz
    if "gunluk_adim_sayisi" in df.columns:
        df["adim_sayisi_bin"] = df["gunluk_adim_sayisi"] / 1000

    # Nabız ve stres birlikte yük göstergesi olabilir
    if {"dinlenik_nabiz_bpm", "stres_skoru"}.issubset(df.columns):
        df["nabiz_stres_yuku"] = (
            df["dinlenik_nabiz_bpm"] * df["stres_skoru"]
        )

    # BMI kategorisi
    if "vucut_kitle_indeksi" in df.columns:
        df["bmi_kategori"] = pd.cut(
            df["vucut_kitle_indeksi"],
            bins=[0, 18.5, 25, 30, np.inf],
            labels=["zayif", "normal", "kilolu", "obez"]
        ).astype("object")

    # Hafta sonu flag
    if "gun_tipi" in df.columns:
        df["hafta_sonu_flag"] = (df["gun_tipi"] == "Hafta sonu").astype(int)

    # Ruh sağlığı ordinal risk skoru
    if "ruh_sagligi_durumu" in df.columns:
        risk_map = {
            "Saglikli": 0,
            "Anksiyete": 1,
            "Depresyon": 2,
            "Anksiyete ve depresyon": 3,
        }

        df["ruh_sagligi_risk_skoru"] = df["ruh_sagligi_durumu"].map(risk_map)

    return df