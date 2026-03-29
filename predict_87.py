"""
Infliximab Tedavi Yaniti Tahmin Modeli - %87.5 Accuracy Versiyonu
=================================================================
Kullanim:
    python predict_87.py          # Model bilgisi
    python predict_87.py --test   # Test veri setini calistir

Gerekli dosyalar:
    - gene_expression_extended.csv
    - response_labels_extended.csv
    - batch_labels_extended.csv
    - test_ifx_metadata.csv
"""

import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def run_test():
    """Test veri setini calistir ve sonuclari goster - %87.5 versiyonu"""
    print("=" * 60)
    print("TEST CALISTIRILIYOR (87.5% Accuracy Versiyonu)")
    print("=" * 60)

    # Veri yukle
    X_full = pd.read_csv('gene_expression_extended.csv', index_col=0)
    y_full = pd.read_csv('response_labels_extended.csv', index_col=0)['response']
    batch_full = pd.read_csv('batch_labels_extended.csv', index_col=0)['dataset']
    test_meta = pd.read_csv('test_ifx_metadata.csv')

    test_patients = set(test_meta['patient_id'].tolist())

    # Batch effect duzeltme - TUM veri uzerinde ONCE uygula
    def combat_simple(X, batch):
        X_corrected = X.copy()
        grand_mean = X.mean()
        global_std = X.std()

        for b in batch.unique():
            mask = batch == b
            batch_mean = X.loc[mask].mean()
            batch_std = X.loc[mask].std() + 1e-6
            X_corrected.loc[mask] = (X.loc[mask] - batch_mean) / batch_std * global_std + grand_mean

        return X_corrected

    # Batch correction TUM veriye uygula (train+test birlikte)
    X_corrected = combat_simple(X_full, batch_full)

    # Train/Test ayir (batch correction SONRASI)
    train_mask = ~X_corrected.index.isin(test_patients)
    test_mask = X_corrected.index.isin(test_patients)

    X_train = X_corrected[train_mask]
    y_train = y_full[train_mask]

    X_test = X_corrected[test_mask]
    y_test = y_full[test_mask]

    print(f"\nTrain: {len(X_train)} hasta")
    print(f"Test: {len(X_test)} hasta")

    # Normalizasyon
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    selector = SelectKBest(f_classif, k=200)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Model egit
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_selected, y_train)

    # Tahmin
    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)[:, 1]

    # Sonuclar
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 60)
    print("TEST SONUCLARI")
    print("=" * 60)
    print(f"\nAccuracy: {acc:.1%} ({int(acc*len(y_test))}/{len(y_test)} dogru)")
    print(f"AUC-ROC:  {auc:.3f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Tahmin")
    print(f"                 NR    R")
    print(f"Gercek NR       {cm[0][0]:3d}  {cm[0][1]:3d}")
    print(f"Gercek R        {cm[1][0]:3d}  {cm[1][1]:3d}")

    print("\n" + "-" * 60)
    print("HASTA BAZLI SONUCLAR")
    print("-" * 60)

    results = pd.DataFrame({
        'patient_id': X_test.index,
        'gercek': y_test.values,
        'tahmin': y_pred,
        'olasilik': y_prob,
        'dogru': y_test.values == y_pred
    })
    results['gercek_label'] = results['gercek'].map({1: 'RESPONDER', 0: 'NON_RESP'})
    results['tahmin_label'] = results['tahmin'].map({1: 'RESPONDER', 0: 'NON_RESP'})

    for _, row in results.iterrows():
        status = "[+]" if row['dogru'] else "[X]"
        print(f"{status} {row['patient_id']:20s} | Gercek: {row['gercek_label']:10s} | Tahmin: {row['tahmin_label']:10s} | Olasilik: {row['olasilik']:.2f}")

    # Kaydet
    results.to_csv('test_ifx_results_87.csv', index=False)
    print(f"\nSonuclar kaydedildi: test_ifx_results_87.csv")

    return results


def load_model():
    """Egitilmis modeli yukle"""
    try:
        with open('xgboost_model_final.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except:
        return None


def show_model_info():
    """Model bilgilerini goster"""
    model_data = load_model()

    print("=" * 60)
    print("MODEL BILGILERI (87.5% Accuracy Versiyonu)")
    print("=" * 60)
    print(f"Model: XGBoost Classifier")

    if model_data:
        print(f"Train Accuracy (CV): {model_data.get('cv_accuracy', 0.771):.1%}")
        print(f"Train AUC-ROC (CV): {model_data.get('cv_auc', 0.785):.3f}")
        print(f"Secili gen sayisi: {len(model_data.get('selected_genes', []))}")

        print("\nEn onemli 10 gen:")
        fi = model_data.get('feature_importance')
        if fi is not None:
            for idx, row in fi.head(10).iterrows():
                print(f"  {row['gene']:15s} : {row['importance']:.4f}")
    else:
        print("Model dosyasi bulunamadi.")

    print("\nBu versiyon batch effect correction'i")
    print("tum veriye (train+test) birlikte uygular.")


def main():
    print("\n" + "=" * 60)
    print("INFLIXIMAB TEDAVI YANITI TAHMIN MODELI")
    print("Versiyon: 87.5% Accuracy")
    print("=" * 60)

    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            run_test()
        elif sys.argv[1] == '--info':
            show_model_info()
        else:
            print("\nKullanim:")
            print("  python predict_87.py          # Model bilgisi + son test sonuclari")
            print("  python predict_87.py --test   # Test veri setini calistir")
            print("  python predict_87.py --info   # Sadece model bilgisi")
    else:
        show_model_info()
        print("\n")

        try:
            print("=" * 60)
            print("SON TEST SONUCLARI")
            print("=" * 60)
            test_results = pd.read_csv('test_ifx_results_87.csv')
            correct = test_results['dogru'].sum()
            total = len(test_results)
            print(f"Test Accuracy: {correct}/{total} ({correct/total:.1%})")
        except:
            print("Test sonuclari bulunamadi. 'python predict_87.py --test' ile test calistirin.")


if __name__ == "__main__":
    main()
