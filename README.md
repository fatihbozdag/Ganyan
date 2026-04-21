# 🏇 Ganyan — TJK Yarış Tahmin Sistemi

Türkiye Jokey Kulübü (TJK) yarış verilerini kazır, LightGBM
LambdaRank tabanlı bir sıralama modeliyle tahmin üretir ve egzotik
(Üçlü / İkili / Sıralı İkili) havuzlar için **backtest'i pozitif**
olan bahis stratejilerini otomatik olarak öneri defterine işler.

Üçlü Top-1 stratejisi için strict out-of-sample backtest (2026-01-16
→ 2026-04-19, 1.477 yarış): **+149.5% ROI**. Canlı defter (3.185
işlemli pick): **+50.9% ROI**. Detaylar aşağıda — lütfen önce
"Dürüst Uyarılar" kısmını okuyun.

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
  rest fitness, class, AGF, soy, ekipman değişikliği vb.), `ranker.py`
  (AGF-farkında LightGBM LambdaRank), `value_model.py` (AGF-kör varyant),
  `bayesian.py` (referans Bayes modeli), `exotics.py` (Harville joint
  probabilities), `picks.py` (strateji-bazlı öneri defteri + grading).
- **web/** + **cli/** — Flask dashboard (HTMX + Bootstrap 5, Türkçe
  arayüz) ve Typer CLI doğrudan predictor/scraper'ı tüketir.

## 🎯 Tahmin Faktörleri

| Kategori | Özellikler |
|---|---|
| Form | Son 20 yarış performansı (S20), KGS (koşmama gün sayısı, 14-28 optimal), form cycle (exponential decay) |
| Hız | En iyi derece (EİD, saniyeye çevrilmiş), speed figure |
| Fiziksel | Kilo farkı, ekipman değişikliği sinyalleri |
| Pazar | AGF (Ağırlıklı Galibiyet Faktörü), Ganyan oranı, son 800m pace |
| Soy | Sire-level win rate ve zemin uyumu |
| Koşu | HP (handikap), yarış sınıfı, pist tipi, GNY |

Her özellik için bkz. `src/ganyan/predictor/features.py`.

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
uv run ganyan predict <race_id>                 # tek yarış
uv run ganyan predict --today --json            # tüm günün tahminleri
uv run ganyan uclu-picks --date 2026-04-21      # Üçlü Top-1 önerileri
uv run ganyan picks --grade                     # bekleyen pick'leri grade et
uv run ganyan picks --since 2026-04-01          # canlı ROI defteri
uv run ganyan exotics-backtest --from 2026-01-16  --model ml  # backtest
uv run ganyan scrape --today                    # bugünün programı
uv run ganyan scrape --results                  # sonuçlar
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

# İlk kazımayı yap (geriye dönük 14 gün önerilir)
uv run ganyan scrape --backfill --from 2026-04-07

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

Her yarış için 4 strateji kaydedilir:

| Strateji | Stake | Ne | Backtest ROI | Canlı ROI |
|---|---|---|---|---|
| `ganyan_top1` | 100 TL | **Referans** — Ganyan Top-1 (modelin favorisi) | takeout'tan ~-20% | ~-20% |
| `sirali_ikili_top1` | 100 TL | Harville Sıralı İkili Top-1 | marginal | -9.9% |
| `uclu_top1` | 100 TL | Harville Üçlü Top-1 | **+149.5%** | **+50.9%** |
| `uclu_box6` | 600 TL | Aynı top-3'ün 6 kutu permütasyonu | **+112.6%** | **+41.5%** |

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
├── predictor/    # features.py, ranker.py, value_model.py,
│                 # exotics.py, picks.py, bayesian.py
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

⚠️ **Not:** Bu sistem sadece eğitim ve araştırma amaçlıdır.
Kumar bağımlılık yapar. Yalnızca kaybetmeyi göze aldığınız tutarları
riske atın.
