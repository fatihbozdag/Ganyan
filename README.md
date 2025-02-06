# At Yarışı Tahmin Sistemi

## Genel Bakış
Bu proje, Türkiye Jokey Kulübü (TJK) yarış verilerini analiz ederek at yarışı sonuçlarını tahmin etmeye yönelik geliştirilmiş bir sistemdir. Makine öğrenimi ve Bayesian analiz yöntemlerini kullanarak yarış sonuçlarını değerlendirir ve tahminler üretir.

## Özellikler
- Web tabanlı kullanıcı arayüzü
- Gerçek zamanlı veri kazıma (TJK websitesinden)
- Makine öğrenimi tabanlı tahminler
- Bayesian olasılık hesaplamaları
- Kombine tahmin sistemi
- Detaylı yarış ve at istatistikleri
- CSV ve SQLite veritabanı desteği

## Sistem Gereksinimleri
- Python 3.x
- SQLite3
- Web tarayıcısı (Chrome, Safari, Firefox vb.)
- İnternet bağlantısı

## Kurulum
1. Projeyi klonlayın:
```bash
git clone https://github.com/fatihbozdag/Ganyan.git
cd Ganyan
```

2. Sanal ortam oluşturun ve aktif edin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac için
venv\Scripts\activate     # Windows için
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Veritabanını oluşturun:
```bash
python scripts/create_db_from_processed.py
```

## Kullanım
1. Web uygulamasını başlatın:
```bash
python app.py
```

2. Tarayıcınızda şu adresi açın:
```
http://localhost:5003
```

### Yarış Verisi Ekleme
1. Ana sayfada "Yeni Yarış Ekle" butonuna tıklayın
2. Yarış bilgilerini girin:
   - Şehir
   - Hipodrom
   - Yarış saati
   - Mesafe
   - Pist tipi
3. At bilgilerini ekleyin:
   - At adı
   - Yaş
   - Kilo
   - Jokey
   - Start pozisyonu
   - HP (Handikap Puanı)
   - Son 6 yarış
   - KGS (Koşmama Gün Sayısı)
   - S20 (Son 20 yarış performansı)
   - EİD (En iyi derece)
   - GNY (Günlük Nispi Yarış puanı)
   - AGF (Ağırlıklı Galibiyet Faktörü)

### Tahmin Görüntüleme
- ML Tahminleri: Makine öğrenimi bazlı tahminler
- Bayesian Tahminler: Olasılık bazlı tahminler
- Kombine Tahminler: İki sistemin birleştirilmiş sonuçları

## Proje Yapısı
```
ganyan/
├── analysis/           # Analiz araçları ve raporlar
├── data/              # Veri dosyaları
│   ├── raw/           # Ham yarış verileri
│   └── processed/     # İşlenmiş veriler
├── scrapers/          # Veri kazıma araçları
├── scripts/           # Yardımcı scriptler
├── templates/         # Web arayüzü şablonları
└── utils/             # Yardımcı fonksiyonlar
```

## Veri Kaynakları
- TJK resmi websitesi (www.tjk.org)
- Geçmiş yarış sonuçları
- At performans verileri
- Hipodrom ve pist bilgileri

## Tahmin Faktörleri
1. Temel Faktörler:
   - At performans geçmişi
   - Jokey performansı
   - Mesafe uyumu
   - Pist tipi uyumu
   - Handikap değerlendirmesi

2. İstatistiksel Faktörler:
   - Kazanma oranı
   - Derece yapma oranı
   - Form durumu
   - Yarış sınıfı uyumu

3. Özel Faktörler:
   - Hava ve pist durumu
   - Yarış tipi
   - Rakip analizi
   - Antrenör faktörü

## Hata Ayıklama
Yaygın hatalar ve çözümleri:
1. Veritabanı Hataları:
   ```bash
   python scripts/create_db_from_processed.py --reset
   ```

2. Veri Kazıma Hataları:
   ```bash
   python scripts/run_scraper.py --debug
   ```

3. Port Çakışması:
   - app.py dosyasında port numarasını değiştirin
   - Mevcut process'leri sonlandırın

## Katkıda Bulunma
1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: XYZ'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## Lisans
Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## İletişim
- GitHub: [@fatihbozdag](https://github.com/fatihbozdag)

## Teşekkürler
Bu projeye katkıda bulunan herkese teşekkürler.

---
**Not**: Bu sistem sadece eğitim ve araştırma amaçlıdır. Gerçek bahis oyunları için kullanılması tavsiye edilmez.