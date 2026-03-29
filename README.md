# Romato - Infliximab Tedavi Yanıtı Tahmin Sistemi

Romatoid Artrit hastalarının **Infliximab** ilacına vereceği yanıtı, tedavi öncesi gen ifade profillerine bakarak tahmin eden bir makine öğrenmesi projesi.

## Ne İşe Yarar?

Romatoid Artrit tedavisinde kullanılan Infliximab ilacı her hastada aynı etkiyi göstermiyor. Bazı hastalar tedaviye yanıt verirken (**Responder**), bazıları vermiyor (**Non-Responder**).

Bu proje, hastanın tedaviye başlamadan önce alınan kan örneğindeki gen ifade verilerine bakarak:
- Hastanın tedaviye yanıt verip vermeyeceğini tahmin eder
- Gereksiz tedavi denemelerinin önüne geçer
- Kişiselleştirilmiş tedavi planlamasına yardımcı olur

## Model Performansı

| Metrik | Değer |
|--------|-------|
| Test Doğruluğu | **%87.5** |
| AUC-ROC | 0.85 |
| Kullanılan Gen Sayısı | 200 |
| Eğitim Hasta Sayısı | 96 |
| Test Hasta Sayısı | 16 |

## Nasıl Çalışır?

1. Hastanın tedavi öncesi gen ifade verisi alınır
2. Batch effect düzeltmesi uygulanır
3. En önemli 200 gen seçilir
4. XGBoost modeli ile tahmin yapılır
5. Sonuç: **Responder** veya **Non-Responder**

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `predict_87.py` | Ana tahmin modeli |
| `demo_page.py` | Demo HTML sayfası oluşturucu |
| `demo_sonuclari.html` | Örnek tahmin sonuçları sayfası |
| `gene_expression_extended.csv` | Gen ifade verileri |
| `response_labels_extended.csv` | Tedavi yanıtı etiketleri |
| `test_ifx_metadata.csv` | Test hastası bilgileri |
| `xgboost_model_final.pkl` | Eğitilmiş model |

## Kurulum

```bash
# Repoyu klonla
git clone https://github.com/KULLANICI_ADI/Romato.git
cd Romato

# Gereksinimleri yükle
pip install -r requirements.txt
```

## Kullanım

### Model Bilgisi
```bash
python predict_87.py
```

### Test Çalıştır
```bash
python predict_87.py --test
```

### Demo Sayfası Oluştur
```bash
python demo_page.py
```
Ardından `demo_sonuclari.html` dosyasını tarayıcıda açın.

### Örnek 2 HASTA İÇİN TEST SONUÇLARI
<img width="1144" height="416" alt="image" src="https://github.com/user-attachments/assets/10ec645d-88a6-4c6a-bcda-2ae64480c97b" />
<img width="1004" height="869" alt="image" src="https://github.com/user-attachments/assets/8cde968b-105e-4520-aedb-3a6893fc1475" />
<img width="1000" height="864" alt="image" src="https://github.com/user-attachments/assets/db42a069-c3f8-4642-97be-cd263ba137f9" />
<img width="1020" height="271" alt="image" src="https://github.com/user-attachments/assets/b44f5d08-e074-4002-ae22-070ffe380098" />





## Veri Kaynakları

Projede kullanılan veriler GEO (Gene Expression Omnibus) veritabanından alınmıştır:
- GSE58795
- GSE78068

## Teknolojiler

- Python 3
- XGBoost
- Scikit-learn
- Pandas / NumPy

## Örnek Sonuç

```
[+] 58795_Subject 407  | Gerçek: RESPONDER     | Tahmin: RESPONDER     | Güven: 97.8%
[+] 78068_35           | Gerçek: NON_RESPONDER | Tahmin: NON_RESPONDER | Güven: 72.4%
```

## Lisans

MIT License
