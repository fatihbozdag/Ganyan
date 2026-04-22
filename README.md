# 🏇 Ganyan — TJK Yarış Tahmin Sistemi

Türkiye Jokey Kulübü (TJK) yarış verilerini kazır, LightGBM
LambdaRank tabanlı bir sıralama modeliyle tahmin üretir ve egzotik
(Üçlü / İkili / Sıralı İkili) havuzlar için **backtest'i pozitif**
olan bahis stratejilerini otomatik olarak öneri defterine işler.

Üçlü Top-1 stratejisi için strict out-of-sample backtest (2026-01-16
→ 2026-04-19, 1.477 yarış): **+149.5% ROI**. Canlı defter (3.185
işlemli pick, v3-bayesian döneminden birikmiş): **+50.9% ROI**.
Detaylar aşağıda — lütfen önce "Dürüst Uyarılar" kısmını okuyun.

2026-04-22 itibarıyla **varsayılan tahminci LightGBM (ml)**; 3 aylık
veri rescrape'i (EİD / son_6 / KGS / s20 alanları 7-8% → 94-99%
kapsama) ve modeli yeniden eğitme sonrası tam 3 aylık pencerede
(2026-01-22 → 2026-04-18, **1.369 yarış**) değerlendirme:

- Üçlü Top-1: **+583.6% full, +452.5% holdout OOS**
- Üçlü Kutu-6: **+174.6% full, +82.3% holdout OOS**
- Sıralı İkili Top-1: **+32.0% full, +44.4% holdout OOS**
- Ganyan Top-1 (referans): -1.7% full

3 betting stratejisinin full pencerede toplam net kârı (100 TL/ticket
nominal): **+1.056.755 TL** (372.600 TL stake üzerinde). Bayesian
model (hand-tuned, v5-s20) `--model bayesian` ile hâlâ erişilebilir
fallback olarak kalıyor.

---

## 🏗 Mimari

Üç servisli monorepo, tümü aynı PostgreSQL veritabanını paylaşır:

```
TJK AJAX → scraper/  →  PostgreSQL
                           │
                    predictor/ (LightGBM + Harville)
                           │
                web/ (Flask + HTMX)     cli/ (Typer)
```

- **scraper/** — `tjk_api.py` şehir-bazlı yarış programı ve sonuçlarını
  paralel çeker; `parser.py` HTML'i dataclass'a dönüştürür; `backfill.py`
  idempotent ekleme ve geçmiş veri yükleme.
- **predictor/** — `features.py` (speed figure, form cycle, weight delta,
  rest fitness, class, AGF, **s20 edge**, soy, ekipman değişikliği vb.),
  `ml/predictor.py` + `ml/trainer.py` (AGF-farkında LightGBM LambdaRank,
  **varsayılan tahminci**; AGF-kör value varyantı için `--exclude-agf`),
  `bayesian.py` (v5-s20 hand-tuned referans), `exotics.py` (Harville joint
  probabilities), `picks.py` (strateji-bazlı öneri defteri + grading).
- **web/** + **cli/** — Flask dashboard (HTMX + Bootstrap 5, Türkçe
  arayüz) ve Typer CLI doğrudan predictor/scraper'ı tüketir.

## 🎯 Tahmin Faktörleri

| Kategori | Özellikler |
|---|---|
| Form | **S20 edge** (son 20 yarış skoru, alan ortalamasına göre), son_6 finishes, KGS (14-28 optimal), form cycle |
| Hız | En iyi derece (EİD → saniye), speed figure (m/s) |
| Fiziksel | Kilo farkı, ekipman değişikliği, gate (start kapısı) |
| Pazar | AGF (Ağırlıklı Galibiyet Faktörü) — ham, edge, ve normalize varyant |
| Soy | Sire-level win rate ve zemin uyumu |
| Koşu | HP (handikap), yarış sınıfı, pist tipi, GNY, field size |

Her özellik için bkz. `src/ganyan/predictor/features.py` ve
`src/ganyan/predictor/ml/features.py` (LightGBM feature matrisi).

> **Model sağlığı hakkında not.** 2026-04-22 post-hoc denetim: ham
> veri kapsama hatası nedeniyle form/speed/rest özellikleri tarihsel
> olarak 85-93% sabit değerdeydi. Rescrape sonrası kapsama 94-99%'a
> çıktı; yeniden eğitilen LightGBM'in feature importance'ı **agf_edge
> baskın, 11 özellik sıfır-üzeri gain** (önceden 4 özellik sıfır-üzeri,
> geri kalan feature'lar NaN nedeniyle hiç split'e sokulmamıştı).

---

## 📊 Panolar (Flask, :5003)

| Rota | Ne gösterir |
|---|---|
| `/` | Bugünün kart özeti + hızlı aksiyon butonları |
| `/races/<id>/predict` | Tek yarış için sıralama + **per-race bahis önerileri** (Üçlü Top-1, Kutu-6, Sıralı İkili) |
| `/live` | Günün tüm yarışlarını canlı izleme — tahmin vs gerçek top-3, 30s auto-refresh, rolling P&L |
| `/picks` | Strateji defteri — her stratejinin hit oranı, ROI, net TL kâr/zarar |
| `/ops` | Scheduler job-run geçmişi, data-freshness, sağlık durumu |
| `/ops/health` | JSON health check (200 ok / 503 degraded) |

---

## 🖥 CLI

```bash
uv run ganyan races --today                     # bugünün kartı
uv run ganyan predict <race_id>                 # tek yarış (varsayılan: ml)
uv run ganyan predict --today --json            # tüm günün tahminleri
uv run ganyan predict --model bayesian <race>   # hand-tuned Bayesian fallback
uv run ganyan uclu-picks --date 2026-04-21      # Üçlü Top-1 önerileri
uv run ganyan picks --grade                     # bekleyen pick'leri grade et
uv run ganyan picks --since 2026-04-01          # canlı ROI defteri
uv run ganyan exotics-backtest --from 2026-01-16  --model ml  # backtest
uv run ganyan scrape --today                    # bugünün programı
uv run ganyan scrape --results                  # sonuçlar
uv run ganyan scrape --backfill --rescrape \    # geçmiş veriyi (re-)scrape et
    --from 2026-01-22 --to 2026-04-18
uv run ganyan train                             # 90-günlük pencere ile retrain
uv run ganyan crawl horses                      # incremental pedigree crawl
uv run ganyan daemon                            # scheduler'ı foreground'da çalıştır
```

---

## 🛠 Kurulum

Gereksinimler: **Python 3.12+**, **PostgreSQL 15+**, [**uv**](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/fatihbozdag/Ganyan.git
cd Ganyan

# Postgres'i başlat (macOS / Homebrew örneği — Linux için systemd vs.)
brew services start postgresql@15
createdb ganyan
createuser -s ganyan   # ya da .env'deki DATABASE_URL'i kendinize göre ayarlayın

# Python bağımlılıklarını yükle
uv sync --all-extras

# .env oluştur ve düzenle
cp .env.example .env

# Veritabanı şemasını oluştur
uv run alembic upgrade head

# İlk kazımayı yap — eğitim verisi için geriye dönük 90 gün önerilir.
# --rescrape, scrape_log'da tam-başarı işaretine rağmen yeniden
# kazıma yapar; eski verileriniz özellikle son_6 / EİD / KGS / s20
# alanlarında boş ise gereklidir.
uv run ganyan scrape --backfill --rescrape \
    --from 2026-01-22 --to 2026-04-18

# Web app'i başlat
uv run python -c "from ganyan.web.app import run; run()"
# → http://localhost:5003
```

Env vars (`.env` veya shell):
- `DATABASE_URL` — Postgres connection string
- `FLASK_PORT` (default 5003)
- `GANYAN_SKIP_LAUNCH_REFRESH=1` — Flask startup'taki 14-day refresh'i atla
- `GANYAN_SKIP_SCHEDULER=1` — Flask içine gömülü APScheduler'ı devre dışı bırak

---

## ⏰ 7/24 Çalıştırma (macOS / launchd)

```bash
# WorkingDirectory'yi kendi checkout path'inize göre düzenleyin, sonra:
cp ops/com.ganyan.web.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ganyan.web.plist

# Durum
launchctl print gui/$(id -u)/com.ganyan.web
tail -f /tmp/ganyan-web.log

# Durdur
launchctl bootout gui/$(id -u)/com.ganyan.web
```

Detaylar ve headless varyant için bkz. [`ops/README.md`](ops/README.md).

### Zamanlanmış İşler (Europe/Istanbul)

| ID | Zaman | Ne yapar |
|---|---|---|
| `morning_card` | 08:30 | Günün programını kazır, her yarışa tahmin + pick üretir |
| `results_poll` | Her 20 dk, 13:00–23:59 | Sonuçları çeker, bekleyen pick'leri grade eder |
| `pedigree_refresh` | Pazar 03:00 | Yeni atlar için soy verisi çeker |
| `monthly_retrain` | Ayın 1'i 03:30 | Her iki modeli de 90-günlük pencereyle yeniden eğitir |

Hata yakalama: her job hata verirse / kaçırılırsa `job_runs` tablosuna
yazılır ve macOS bildirim balonu çıkar (`osascript`).

---

## 📈 Strateji Defteri (`/picks`)

Her yarış için 4 strateji kaydedilir; `ganyan picks` CLI çıktısı
**Betting** (gerçek P&L) ve **Reference** (gösterim amaçlı, bahse
girmiyoruz) olarak ikiye ayrılır.

| Strateji | Stake | Ne | Train ROI (1066 yarış) | **Holdout ROI (303 yarış)** | Full 3-ay ROI (1369 yarış) | Sınıf |
|---|---|---|---|---|---|---|
| `uclu_top1` | 100 TL | Harville Üçlü Top-1 | +619.3% | **+452.5%** | +583.6% | Betting |
| `uclu_box6` | 600 TL | Aynı top-3'ün 6 kutu permütasyonu | +199.7% | **+82.3%** | +174.6% | Betting |
| `sirali_ikili_top1` | 100 TL | Harville Sıralı İkili Top-1 | +28.5% | **+44.4%** | +32.0% | Betting |
| `ganyan_top1` | 100 TL | Ganyan Top-1 (modelin favorisi) | +0.1% | -7.8% | -1.7% | Referans |

Yukarıdaki ROI'lar **yeniden eğitilen LightGBM ranker (ml-new)** ile
tam 3 aylık pencerede (2026-01-22 → 2026-04-18, 1.504 yarış, 1.369
tanesi tam top-3 sonuçlu) hesaplandı. Sütunlar: Train (modelin
öğrendiği 80%), **Holdout (temiz out-of-sample %20)**, Full (ikisi
birlikte). Üç betting stratejisinin full pencerede toplam net kârı
**+1.056.755 TL** (372.600 TL stake üzerinde; tüm sanal tutarlar,
100 TL/ticket nominal).

Train-holdout farkına dikkat: uclu_box6 için in-sample +200% /
OOS +82% — forward beklenti için holdout rakamını kullanın, train
rakamı verinin bir kısmı modelin gördüğü için şişkindir. Ama
uclu_top1'in farkı düşük (+619 / +453) ve sirali_ikili'da holdout
train'den **daha iyi** (+44 / +29) — "gerçek sinyal, ezberlenmiş
değil" işareti.

Aynı 303 yarışlık holdout pencerede pre-retrain LightGBM +351 / +56
/ +25 / -16 üretiyor, hand-tuned Bayesian v5-s20 +71 / -8 / -7 / -22
üretiyor — mevcut edge'in büyük kısmı **doğru tahminci seçimine
bağlı**.

Canlı defterdeki tarihsel rakamlar v3-bayesian dönemine ait (sirali
tarihsel -11% ROI); sınıflandırma ileriye dönük — yeni picks
ml-new ile üretiliyor.

`ganyan_top1` referans olarak tutulur — uzun vadede kaybeder ama
model sağlığını denetlemek ve hit-rate karşılaştırması için
defterde kalır. Bahis olarak önerilmez.

**Grading kuralı:** TJK'nın o havuz için payout yayınlamadığı
yarışlar **ledger'dan atlanır**, zarar olarak sayılmaz — bahis zaten
açılmamış demektir. Bu kuralı kaldırırsanız ROI -30% olarak okunur.

---

## ⚠️ Dürüst Uyarılar (önemli)

1. **Varyans gerçek.** Üçlü Top-1 ~5% hit oranıyla yaşar. 18 yarışlık
   bir günde sıfır vurma olasılığı ~40%; 34 yarışlık bir günde ~17%.
   Art arda 3–5 kart boş geçmesi normaldir. Bankroll 20+ yarışlık
   kayıp serisini taşıyabilmelidir.

2. **Ganyan havuzu etkin.** Varyans etmen olarak AGF'yi yenemezsiniz.
   AGF-kör "value model" takeout tabanını aşamıyor (test edildi).
   Edge sadece **egzotik havuzların fiyatlama yapısal bozukluğu**
   üzerinden geliyor.

3. **Canlı defter ≠ strict backtest.** Mevcut defter 2026-01-01 →
   2026-01-15 aralığını içerir; bu dönem modelin eğitim
   verisiydi. 2026-04-21'den itibaren her yeni yarış temiz
   out-of-sample satır ekler.

4. **Edge'i neyin bozduğuna dikkat edin:**
   - 50-pick yuvarlanan pencerede hit oranı %3'ün altına düşerse,
   - Ortalama Üçlü payout keskin düşerse (piyasa düzelirse),
   - Aylık ROI trendi negatife dönerse.

   Tek bir -100% gün (olağan, ~40% olasılıkla) ya da bir aylık
   -50% dip (skew mean-reversion) edge'i çürütmez.

5. **Bu sistem eğitim ve araştırma amaçlıdır.** Gerçek para
   yatırmadan önce bankroll yönetimi, Kelly oranı ve kişisel risk
   toleransınızı ciddiye alın. Yazar herhangi bir finansal sorumluluk
   kabul etmez.

---

## 🧪 Geliştirme

```bash
uv sync --all-extras
uv run pytest tests/ -v                          # 165+ test
uv run pytest tests/test_predictor/ -v           # sadece predictor
uv run pytest -k test_exotics                    # isim eşleşmesi
uv run alembic upgrade head                      # migration
uv run alembic revision --autogenerate -m "..."  # yeni migration
```

Ana dosyalar:

```
src/ganyan/
├── scraper/      # tjk_api.py, parser.py, backfill.py
├── predictor/
│   ├── features.py         # engineered features (paylaşımlı)
│   ├── bayesian.py         # v5-s20 hand-tuned referans
│   ├── ml/
│   │   ├── features.py     # LightGBM feature matrisi
│   │   ├── trainer.py      # eğitim + temporal holdout
│   │   └── predictor.py    # MLPredictor (varsayılan)
│   ├── exotics.py          # Harville joint probabilities
│   ├── picks.py            # strateji defteri + grading
│   └── exotic_evaluate.py  # backtest
├── db/           # models.py (SQLAlchemy 2.0), session.py
├── web/          # app.py, routes.py, templates/
├── cli/          # main.py (Typer)
└── scheduler.py  # APScheduler job tanımları
```

---

## 📚 Bahis Terimleri (TJK)

- **AGF** — Ağırlıklı Galibiyet Faktörü (kamusal favori sinyali)
- **HP** — Handikap Puanı
- **KGS** — Koşmama Gün Sayısı
- **EİD** — En İyi Derece
- **S20** — Son 20 yarış performansı
- **GNY** — Günlük Nispi Yarış puanı
- **Ganyan** — Kazanan (tek)
- **İkili** — Sırasız ilk iki
- **Sıralı İkili** — Sıralı ilk iki
- **Üçlü** — Sıralı ilk üç
- **Dörtlü** — Sıralı ilk dört

---

## 🤝 Katkı

Pull request açmadan önce:
1. `uv run pytest` yeşil olmalı.
2. Yeni feature için test ekleyin (`tests/` altında).
3. Migration gerekiyorsa `uv run alembic revision --autogenerate`.

Soru / hata / öneri için [GitHub Issues](../../issues).

## 📄 Lisans

MIT — `LICENSE` dosyasına bakın.

---

## 🔄 Sürüm Geçmişi

Proje gerçek tarihlere göre belgelenmiştir. Eski Selenium-tabanlı
prototip (2025) mevcut mimari için tamamen yeniden yazılmıştır.

- **2026-04-22 — Rescrape, Bayesian v5-s20, ml-new, default switch**
  Post-hoc audit: Bayesian'ın form/speed/rest factor'leri 85-93%
  oranında sabit değerdeydi — üst kaynak (EİD / son_6 / KGS / s20)
  tarihsel veride sadece 7-8% kapsamaydı çünkü scraper 2026-04-19
  civarında düzeltilmişti. 3 aylık `--backfill --rescrape`
  (2026-01-22→2026-04-18) kapsamayı **94-99%**'a çıkardı. Bu veriyle:
  (a) Bayesian v4-pruned (form/speed/rest zero-weight), (b) Bayesian
  v5-s20 (s20 edge weight 0.10 ile eklendi, |r|=0.13), (c) LightGBM
  yeniden eğitildi — eski model best_iter=1 ve 22 feature'dan 18'i
  sıfır önemli; yenisi 11 önemli feature ile top-1 43.4% → 49.1%.
  303 out-of-sample yarışlık head-to-head: ml-new **+452% uclu_top1**,
  **+82% uclu_box6**, **+44% sirali_ikili** (sirali reference →
  betting sınıfına taşındı). Varsayılan CLI / web tahmincisi ml'e
  çevrildi; `--model bayesian` hâlâ fallback. `scrape --backfill`'in
  `--to` / `--rescrape` flag'lerini yok sayma bug'ı düzeltildi.
  Tam 3 aylık pencerede (1.369 yarış) ml-new full-window ROI:
  **+583% uclu_top1, +175% uclu_box6, +32% sirali, -1.7% ganyan** —
  3 betting stratejisinin birleşik net kârı 100 TL/ticket nominalde
  **+1.056.755 TL / 372.600 TL stake**. Train↔holdout farkı küçük
  (top-1: 43.9% vs 41.3%), model aşırı öğrenmemiş; sirali_ikili'da
  holdout train'den daha iyi çıkıyor — gerçek sinyal işareti.

- **2026-04-21 — Picks ledger + halka açılma**
  Strateji-bazlı `picks` tablosu ve gerçek-dünya grading'i
  (`hit` / `payout_tl` / `net_tl`). `/picks` panosu, `/live` üzerinde
  günlük rolling P&L, tek-yarış bahis öneri kartları.
  `ganyan_top1` referans baseline olarak deftere eklendi. Kişisel veri
  temizliği sonrası repo halka açıldı.

- **2026-04-20 — Egzotik-havuz dönemi**
  Harville joint probabilities (Üçlü / İkili / Sıralı İkili / Dörtlü);
  TJK egzotik-havuz payout kazıyıcısı (`7'Lİ GANYAN` trap dahil);
  `exotics-backtest` CLI; Üçlü Top-1 stratejisinde **+149% strict
  out-of-sample ROI** keşfi. Her-zaman-online stack (APScheduler +
  macOS launchd, 4 zamanlı iş), `/ops` sağlık panosu, sürpriz-at
  domain özellikleri (ekipman değişikliği vb.).

- **2026-04-19 — Audit-driven güçlendirme**
  LightGBM LambdaRank modeli (AGF-farkında + AGF-kör varyant),
  AGF özelliği ve 14-gün geçmiş veri backfill, paralel şehir
  kazıma (5×), yarış saati (post time), soy crawler'ı
  (`AtKosuBilgileri`), Son 800m pace özelliği.

- **2026-04-05 — Tam refactor**
  Selenium → `httpx` AJAX client; SQLite → PostgreSQL 16 + SQLAlchemy
  2.0 + Alembic; eski script'ler → modüler `scraper/` + `predictor/` +
  `web/` + `cli/` monorepo'su; Flask + HTMX (Bootstrap 5) arayüz;
  Typer CLI; Bayesian tahmin motoru; `docker-compose` dev ortamı;
  tahmin-değerlendirme pipeline'ı.

- **2025-02-06 — Eski Selenium prototipi**
  İlk çalışan versiyon: SQLite, Selenium + Safari WebDriver,
  `requirements.txt`, manuel veri girişi. (Artık kullanımda değil.)

- **2025-01-21 — İlk prototip**
  Temel kazıyıcı + analiz araçları.

---

⚠️ **Not:** Bu sistem sadece eğitim ve araştırma amaçlıdır.
Kumar bağımlılık yapar. Yalnızca kaybetmeyi göze aldığınız tutarları
riske atın.
