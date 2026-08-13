# 🏇 Ganyan — TJK Yarış Tahmin Sistemi

Türkiye Jokey Kulübü (TJK) yarış verilerini kazır, LightGBM
LambdaRank ranker'ı + hiyerarşik Bayesian Plackett-Luce modelden
oluşan iki katmanlı bir tahminciyle olasılık üretir, Harville joint
probabilities ile egzotik bahis kombinasyonlarını puanlar ve günlük
"hangi yarışta oyna, hangisini atla" tavsiyeleri çıkarır.

**Birincil metrik — top-1 isabet oranı**, ROI değil. 1.841 graded
pick'lik canlı defter üzerinde Ganyan top-1 hit %37.6, Sıralı İkili
top-1 %12.4, Üçlü top-1 %3.9, Üçlü Kutu-6 %13.4. ROI rakamları havuz
takeout dinamiklerini yansıtır — model kalitesinin doğrudan ölçüsü
değildir.

**Edge retraction (2026-04-30):** İlk üçlü "+583% ROI" rakamı, grader
sıralı üçlü payoutunu 1 TL birim olarak skorladığı için 2× şişmişti
— gerçek birim 2 TL bilet başına. Operatörün gerçek bileti (192 TL
yatırım → 370 TL ödeme) hatayı ortaya çıkardı; defter geriye dönük
düzeltildi. Düzeltme sonrası gerçek ROI: **uclu_top1 ≈ −32%,
uclu_box6 ≈ −32%, sirali_ikili_top1 ≈ −13%, ganyan_top1 ≈ −20%** —
yani egzotik havuzlarda pozitif edge **yok**, sistem takeout
tabanında. Model hâlâ top-1 doğruluğu için kullanışlı; gerçek para
bahsi tavsiye edilmez.

**Bayes geçit (2026-04-29):** Hierarchical Plackett-Luce posterior'u
(`models/bayes_pl_v3`) belirsiz yarışları atlamak için filtre olarak
çalışır. `ganyan advice` ve `/advice` web rotası sadece Bayes top-1
ortalaması ≥35% **ve** %5 alt sınırı ≥20% olan yarışları geçirir;
tipik bir günde 18 yarıştan 1-3'ü geçit eşiğini aşar. Detaylar
aşağıda — lütfen önce "Dürüst Uyarılar" kısmını okuyun.

---

## 🏗 Mimari

Üç servisli monorepo, tümü aynı PostgreSQL veritabanını paylaşır:

```
TJK AJAX  →  scraper/        →  PostgreSQL
                                    │
                          predictor/ (11-head ensemble)
                          ├─ LightGBM LambdaRank (top-1 ranker)
                          ├─ Hierarchical Bayesian PL (skip-gate)
                          ├─ Harville joint probabilities (egzotik)
                          └─ Cohort filter + trip-wire
                                    │
                          picks/ + advice/ ledger
                                    │
                  web/ (Flask + HTMX)     cli/ (Typer)
```

- **scraper/** — `tjk_api.py` şehir-bazlı yarış programı ve sonuçlarını
  paralel çeker; `parser.py` HTML'i dataclass'a dönüştürür; `backfill.py`
  idempotent ekleme ve geçmiş veri yükleme; `external/` klasöründe
  yarisrehberi tipster, TJK discipline (jokey ceza), workouts (idman
  istatistikleri), track conditions (pist bilgileri), steward reports
  (komiser raporları) için pluggable scraper framework'ü.
- **predictor/** — `features.py` (speed figure, form cycle, weight delta,
  rest fitness, class, AGF, **s20 edge**, soy, ekipman değişikliği vb.),
  `ml/ensemble.py` + `ml/predictor.py` + `ml/trainer.py` (AGF-farkında
  LightGBM LambdaRank ranker, AGF-kör value, ev, finish-time + 7
  spec-class head, toplam 11-head ensemble), `bayes/` (Hierarchical
  Plackett-Luce, ADVI fit ile eğitilmiş; mean-field posterior'u skip-gate
  filtresi olarak çalışır), `exotics.py` (Harville joint probabilities),
  `picks.py` (strateji-bazlı öneri defteri + grading), `trip_wire.py`
  (model konfidansının 90-günlük baseline'a göre ±2σ sapmasını izler;
  asimetrik halt — sadece UNDER-confidence advice üretmeyi durdurur).
- **web/** + **cli/** — Flask dashboard (HTMX + Bootstrap 5, Türkçe
  arayüz) ve Typer CLI doğrudan predictor/scraper'ı tüketir; `/advice`,
  `/picks`, `/live`, `/ops` panoları + `ganyan advice`, `ganyan morning`,
  `ganyan picks`, `ganyan tune-thresholds` komutları.

## 🎯 Tahmin Faktörleri

| Kategori | Özellikler |
|---|---|
| Form | **S20 edge** (son 20 yarış skoru, alan ortalamasına göre), son_6 finishes, KGS (14-28 optimal), form cycle |
| Hız | En iyi derece (EİD → saniye), speed figure (m/s) |
| Fiziksel | Kilo farkı, ekipman değişikliği, gate (start kapısı) |
| Pazar | AGF (Ağırlıklı Galibiyet Faktörü) — ham, edge, ve normalize varyant |
| Soy | Sire & dam win rate, zemin-koşullu varyantlar (sire/dam × surface) |
| Jokey × pist | Jockey'nin spesifik pistteki Bayesian-smoothed win rate |
| Koşu | HP (handikap), yarış sınıfı, pist tipi, GNY, field size |

Her özellik için bkz. `src/ganyan/predictor/features.py` ve
`src/ganyan/predictor/ml/features.py` (LightGBM feature matrisi).

> **Model sağlığı hakkında not.** 2026-04-22 post-hoc denetim: ham
> veri kapsama hatası nedeniyle form/speed/rest özellikleri tarihsel
> olarak 85-93% sabit değerdeydi. Rescrape sonrası kapsama 94-99%'a
> çıktı; yeniden eğitilen LightGBM'in feature importance'ı **agf_edge
> baskın, 11 özellik sıfır-üzeri gain** (önceden 4 özellik sıfır-üzeri,
> geri kalan feature'lar NaN nedeniyle hiç split'e sokulmamıştı).

> **Pedigree v1 — OOS-validated lift (2026-05-06).** İki yeni soy
> feature (`dam_win_rate`, `dam_surface_rate`) ve bir jokey×pist
> etkileşim feature'ı (`jockey_track_win_rate`) eklendi (FEATURE_COLUMNS
> 39 → 42). Live LightGBM ranker yeniden eğitildi.
>
> Out-of-sample test (`logs/oos_pedigree_v1.py`): 2025-01-01 → 2026-02-04
> penceresi (400 gün, 7020 tam-alan resulted yarış, ≥4 entries):
>
> | | Top-1 | Top-3 |
> |---|---|---|
> | Baseline (39 feature) | 33.95% | 67.14% |
> | Pedigree v1 (42 feature) | **35.21%** | **69.05%** |
> | Lift | **+1.27pp** | **+1.91pp** |
>
> V2 retraction'ın çıkardığı OOS barı (≥365 gün AND ≥1500 yarış AND
> top-1 lift ≥ +1pp) ilk denemede aşıldı. V2 features (in-sample +3.6pp,
> OOS +0.07pp) ile karşılaştırıldığında: bu sefer in-sample lift OOS'da
> replicate oluyor — overfit değil, gerçek yeni bilgi (parent çiftinin
> dam tarafı + jokey'nin pist-spesifik formu mevcut feature'lardan
> çıkarılamaz). Feature importance'ta `dam_surface_rate` #5,
> `jockey_track_win_rate` #9 sırada (training holdout).

> **Per-model OOS geçidi + ölçüm düzeltmesi (2026-08-14).**
> `logs/discordance_oos_backtest.py` betiğinin `--model` bayrağını
> sessizce YOK SAYDIĞI keşfedildi (argparse yok; her zaman canlı
> ensemble'ı skorlar) — o yolla üretilmiş tekil-model "geçit" sayıları
> aslında ensemble'ın kendisiyle karşılaştırmasıydı. Çalışan geçit
> artık `logs/oos_model_gate.py`: aynı yarışlar üzerinde eşleştirilmiş
> canlı-vs-aday karşılaştırma + exact McNemar + AGF-favori baseline.
> İlk gerçek koşu (6.939 OOS yarış, 2025-01-01 → 2026-01-30):
>
> | | Top-1 |
> |---|---|
> | Canlı model (pedigree_v1, 42 feature) | **35.22%** |
> | AGF favorisi (saf kalabalık) | 34.50% |
> | Aday v2 (41 leak-fixed feature, 90 günlük pencere) | 33.20% |
>
> Canlı model kalabalığı +0.72pp geçiyor; 90 günlük pencereyle eğitilen
> aday ise kalabalığın ALTINDA (−2.02pp vs canlı, McNemar p≈0) →
> **swap yok**. Ders: 90 günlük eğitim penceresi Şubat–Mayıs
> ağırlıklarıyla yarışamayacak kadar küçük/sezonsal. Temporal-leak
> düzeltmeleri (tarih-kapsamlı scrape sentinelleri, per-date
> `agf_reliability` cutoff'u, ölü `apprentice_jockey` kolonunun
> çıkarılması) model swap'ı için değil pipeline dürüstlüğü için kalıcı.

---

## 📊 Panolar (Flask, :5003)

| Rota | Ne gösterir |
|---|---|
| `/` | Bugünün kart özeti + hızlı aksiyon butonları |
| `/advice` | Bayes-geçit + cohort filtreden geçmiş yarışlar için günlük "ne oynayım" tavsiyesi; Kelly kalibreli stake, Harville breakdown, trip-wire banner |
| `/races/<id>/predict` | Tek yarış için sıralama + **per-race bahis önerileri** (Üçlü Top-1, Kutu-6, Sıralı İkili) |
| `/live` | Günün tüm yarışlarını canlı izleme — tahmin vs gerçek top-3, 30s auto-refresh, rolling P&L |
| `/picks` | Strateji defteri — her stratejinin hit oranı, ROI, net TL kâr/zarar (top-1 hit oranı birincil metrik) |
| `/history` | Geçmiş yarış-bazlı tahmin vs sonuç defteri |
| `/ops` | Scheduler job-run geçmişi, data-freshness, sağlık durumu |
| `/ops/health` | JSON health check (200 ok / 503 degraded) |

---

## 🖥 CLI

```bash
uv run ganyan races --today                     # bugünün kartı
uv run ganyan predict <race_id>                 # tek yarış (varsayılan: ml single-head)
uv run ganyan predict --today --model ensemble  # günün tamamı, 11-head ensemble ile
uv run ganyan predict --today --json            # tüm günün tahminleri
uv run ganyan predictions                       # daemon'ın kaydettiği ensemble tahminleri (read-only)
uv run ganyan advice                            # bugünün Bayes-geçit + Kelly tavsiyeleri
uv run ganyan advice --no-cohort-filter         # cohort filtresini kapat
uv run ganyan advice --bayes-min-prob 0.30      # geçit eşiğini gevşet
uv run ganyan morning                           # tek-atışta scrape + predict + picks
uv run ganyan uclu-picks --date 2026-04-21      # Üçlü Top-1 önerileri
uv run ganyan multi-picks                       # çoklu-yarış (6'lı) paper-trade kuponları
uv run ganyan multi-grade                       # sonuçlanan çoklu-yarış kuponlarını grade et
uv run ganyan picks --grade                     # bekleyen pick'leri grade et
uv run ganyan picks --since 2026-04-01          # canlı ROI defteri
uv run ganyan tune-thresholds                   # ledger üzerinde min-prob arar
uv run ganyan exotics-backtest --from 2026-01-16 --model ml  # backtest
uv run ganyan scrape --today                    # bugünün programı
uv run ganyan scrape --results                  # sonuçlar
uv run ganyan scrape --backfill --rescrape \    # geçmiş veriyi (re-)scrape et
    --from 2026-01-22 --to 2026-04-18
# DİKKAT: çıplak `ganyan train` CANLI modelin üzerine yazar. Aday isme
# eğitin, OOS geçidini (≥ +1pp top-1) geçmeden swap etmeyin:
uv run ganyan train --model-name candidates/lightgbm_ranker_aday
uv run python logs/oos_model_gate.py --candidate candidates/lightgbm_ranker_aday
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
| `morning_card` | 08:30 | Günün programını kazır, tahmin + pick üretir, pist başına günlük 6'lı paper-trade kuponu yazar |
| `external_signals` | 09:15 + 18:15 | Tipster + ceza + workout + pist + komiser plugin'lerini çalıştırır |
| `agf_snapshot` | 11:00–22:30, :00/:30 | AGF zaman serisini kaydeder (late-drift feature'ının ham verisi) |
| `repredict_upcoming` | 11:05–20:35, :05/:35 | Başlamamış yarışları güncel AGF ile yeniden tahmin eder; başlamış yarışların pick'i frozen |
| `results_poll` | 13:00–23:40, her 20 dk | Sonuçları çeker, bekleyen pick'leri ve çoklu-yarış kuponlarını grade eder |
| `pedigree_refresh` | Pazar 03:00 | Yeni atlar için soy verisi çeker |
| `monthly_retrain` | — | **DEVRE DIŞI (2026-08-12).** OOS geçidi olmadan canlı modelin üzerine yazıyordu; 2026-06-01'de fire edip top-1'i 42.9% → 40.9% düşürdü. Aday-isim + `logs/oos_model_gate.py` akışı kurulmadan yeniden açmayın |

Hata yakalama: her job hata verirse / kaçırılırsa `job_runs` tablosuna
yazılır ve macOS bildirim balonu çıkar (`osascript`).

---

## 📈 Strateji Defteri (`/picks`)

Her yarış için 4 strateji kaydedilir. **Birincil metrik top-1 isabet
oranıdır**, ROI değil — payout TJK havuz dinamiklerini (takeout,
"devren" carry-over, retail davranış) yansıtır, model kalitesini değil.

**Canlı defter (1.841 graded pick, post-birim-fix):**

| Strateji | Stake/bilet | Ne | Hit% (top-1) | ROI |
|---|---|---|---|---|
| `ganyan_top1` | 100 TL | Ganyan Top-1 (modelin favorisi) | **37.6%** (693/1841) | −19.9% |
| `sirali_ikili_top1` | 100 TL | Harville Sıralı İkili tek-kombinasyon | 12.4% (229/1841) | −12.6% |
| `uclu_box6` | 600 TL (6 bilet × 100) | Top-3'ün 6 kutu permütasyonu | 13.4% (114/848) | −32.3% |
| `uclu_top1` | 100 TL | Harville Üçlü tek-kombinasyon | 3.9% (33/848) | −32.2% |

Top-1 hit %37.6 model kalitesi açısından sağlıklı bir sinyal — alan
ortalama büyüklüğü ~12 at olduğunda 1/N rastgele baseline %8.3'tür.
Üçlü ve İkili'deki negatif ROI **havuz takeout tabanı** (~%30-35) ile
tutarlı; egzotik havuzlarda fiyatlama bozukluğundan kaynaklanan bir
edge ön çalışmalarda gözlenmişti ama 2026-04-30 birim düzeltmesinden
sonra artifact olduğu anlaşıldı.

> **Birim düzeltmesi (2026-04-30):** TJK'nın `uclu_payout_tl` alanı
> bilet başına 2 TL birim üzerinden ödemeyi yayınlar — TL başına
> değil. Eski grader bu bölmeyi atladığı için Üçlü kazançları 2×
> şişkin görünüyordu. Operatörün gerçek Ankara R8 bileti
> (192 TL → 370 TL) tutarsızlığı ortaya çıkardı; defter geriye dönük
> düzeltildi. Pre-fix README'deki "+583% Üçlü ROI" rakamı geçerli
> değildir.

**Bayes geçit + Kelly (gerçek-dünya advice akışı):**
`ganyan advice` ve `/advice` web rotası **Bayes posterior** üzerinden
yarış filtrelemesi yapar:

- **Cohort filtresi**: Maiden / dişi cohortları, alan ≥ 13, Şanlıurfa
  ve Bursa hipodromları otomatik atlanır (14 günlük backtest +5,870 TL
  bias).
- **Bayes geçit**: Hierarchical PL posterior'unun top-1 ortalaması
  ≥35% **ve** %5 alt sınırı ≥20% olmazsa yarış atlanır.
- **Kelly stake**: `base × (0.5 + 0.5 × confidence)` — sınırda
  yarışlar yarım Kelly, yüksek konfidans tam Kelly.
- **Trip-wire**: Günlük ortalama top-1 model konfidansı 90-günlük
  baseline'dan ±2σ saparsa banner gösterilir; **asimetrik** —
  UNDER-confidence advice üretmeyi durdurur, OVER-confidence sadece
  uyarı verir (over-confidence feature pipeline bozulduğunun işareti
  değil; under-confidence olduğunun işaretidir).

Tipik bir günde 18 yarışlık karttan 1-3'ü geçit eşiğini aşar. Eşiği
gevşetmek için `--bayes-min-prob`, kapatmak için `--no-bayes-skip`.

**Grading kuralı:** TJK'nın o havuz için payout yayınlamadığı
yarışlar **ledger'dan atlanır**, zarar olarak sayılmaz — bahis zaten
açılmamış demektir.

---

## ⚠️ Dürüst Uyarılar (önemli)

1. **Pozitif edge yok (2026-04-30 retraction sonrası).** Önceki
   "+583% Üçlü ROI" rakamı per-bilet birim hatasıydı; düzeltilmiş
   defterde tüm 4 stratejinin ROI'ı negatif. Sistem **model
   doğruluğu** (top-1 hit %37.6) için kullanışlıdır; **gerçek bahis**
   için değil. Model halka açık verilerden ulaşılabilen tüm sinyalleri
   sömürmüş durumda; yapısal tavan ~%43 top-1 olarak görünüyor.

2. **Varyans gerçek.** Üçlü Top-1 ~%4 hit oranıyla yaşar. 18 yarışlık
   bir günde sıfır vurma olasılığı ~%48; 34 yarışlık bir günde ~%24.
   Art arda 3–5 kart boş geçmesi normaldir.

3. **Ganyan havuzu etkin.** Varyans etmen olarak AGF'yi yenemezsiniz.
   AGF-kör "value model" takeout tabanını aşamıyor (test edildi).
   Egzotik havuzlardaki bozukluk hipotezi (uclu_top1 +149%) küçük-N
   pencerelerde gerçek görünmüştü ama daha geniş N + birim
   düzeltmesinde takeout tabanına geri döndü.

4. **Bayes geçit gerçek bir kalite sinyali.** `ganyan advice`'in
   geçirdiği yarışlarda top-1 hit oranı, geçit-dışı yarışların
   ortalamasından belirgin yüksek. Geçit "bahis-değer" değil
   "model-konfidans" filtresi olarak yararlı.

5. **Trip-wire'a güvenin.** Sabah avg top-1 konfidansı 90-günlük
   baseline'dan +2σ'dan fazla saptığında model overconfident demektir
   ve genellikle hit oranı altında düşer (2026-05-01 örneği:
   z=+2.57, model avg %26.3, gerçek hit %23.1 — uyarı doğru çıktı).
   −2σ ise feature pipeline bozulduğunu gösterir → halt.

6. **Bu sistem eğitim ve araştırma amaçlıdır.** Gerçek para
   yatırmadan önce bankroll yönetimi, Kelly oranı ve kişisel risk
   toleransınızı ciddiye alın. Yazar herhangi bir finansal sorumluluk
   kabul etmez.

---

## 🧪 Geliştirme

```bash
uv sync --all-extras
uv run pytest tests/ -v                          # 271 test
uv run pytest tests/test_predictor/ -v           # sadece predictor
uv run pytest -k test_exotics                    # isim eşleşmesi
uv run alembic upgrade head                      # migration
uv run alembic revision --autogenerate -m "..."  # yeni migration
```

Ana dosyalar:

```
src/ganyan/
├── scraper/
│   ├── tjk_api.py, parser.py, backfill.py
│   └── external/           # tipster, discipline, workouts, track-cond, steward
├── predictor/
│   ├── features.py         # engineered features (paylaşımlı)
│   ├── bayesian.py         # v5-s20 hand-tuned referans (legacy fallback)
│   ├── ml/
│   │   ├── features.py     # LightGBM feature matrisi
│   │   ├── trainer.py      # eğitim + temporal holdout
│   │   ├── predictor.py    # MLPredictor (single-head)
│   │   ├── ensemble.py     # 11-head ensemble (varsayılan tahminci)
│   │   └── linear_ranker.py# Plackett-Luce + conditional-logit baselines
│   ├── bayes/              # Hierarchical Bayesian PL (skip-gate)
│   │   ├── trainer.py      # ADVI fit + posterior persist
│   │   └── predictor.py    # credible intervals on top-1 prob
│   ├── exotics.py          # Harville joint probabilities
│   ├── picks.py            # strateji defteri + grading
│   ├── trip_wire.py        # 90-günlük baseline ±2σ asimetrik halt
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

- **2026-08-12/14 — Stack canlandırma + ölçüm dürüstlüğü**
  6 haftalık kesinti giderildi (launchd agent'ları disable olmuş;
  uv'nin macOS TCC/Full-Disk-Access grant'i uv upgrade'iyle düşmüştü)
  ve 45 günlük veri açığı kapatıldı: +825 tam-alan sonuçlu yarış,
  eğitim havuzu 13.4K yarış. `monthly_retrain` DEVRE DIŞI — 2026-06-01'de
  OOS geçidi olmadan canlı modelin üzerine yazıp top-1'i 42.9%→40.9%
  düşürmüştü; canlı ağırlıklar geri yüklendi. Ölü hava-durumu head'i
  (`lightgbm_ranker_wx_v1`, 5 all-NaN feature ile oy kullanıyordu)
  karantinaya alındı → ensemble 11 head. TJK'nın eksik TLS zinciri
  için `truststore` düzeltmesi (temiz clone artık macOS'ta kazıyabiliyor).
  Üç temporal-leak düzeltmesi + çalışan per-model OOS geçidi
  (`logs/oos_model_gate.py`; eski betiğin `--model`'i no-op'tu —
  yukarıdaki not). İlk gerçek geçit koşusu: aday −2.02pp, swap yok.
  `morning_card` artık pist başına günlük 6'lı paper-trade kuponu
  üretiyor (`multi_race_picks` tablosu, `ganyan multi-grade` ile grade).

- **2026-05-01 — Asimetrik trip-wire**
  Trip-wire OVER-confidence durumunda halt ETMİYOR — feature pipeline
  bozulduğunun işareti UNDER-confidence'tır (avg top-1 baseline'dan
  -2σ). Aynı gün ilk testte z=+2.57 over-confident sinyalde model
  avg %26.3 vs gerçek hit %23.1 — uyarı doğru, halt yanlış olurdu.

- **2026-04-30 — Üçlü edge retraction**
  TJK'nın `uclu_payout_tl` alanının bilet başına 2 TL birim üzerinden
  ödeme yayınladığı keşfedildi (operatörün gerçek 192 TL → 370 TL
  Ankara R8 bileti tutarsızlığı ortaya çıkardı). Eski grader 1 TL
  birim varsayıyordu, Üçlü kazançlarını 2× şişiriyordu. Defter
  geriye dönük düzeltildi: uclu_top1 ROI **+583% → −31.9%**, uclu_box6
  **+175% → −32.0%**. Picks.py + grading test'leri yamandı; README
  ve picks panosundaki "Üçlü kâr pozitif" copy retract edildi.
  Operasyonel sonuç: pozitif edge yok, sistem **takeout tabanında**.

- **2026-04-29 — Hierarchical Bayesian Plackett-Luce**
  PyMC 5 + ADVI fit ile hiyerarşik PL modeli (jokey/sire/track-dist
  random effects, AGF informative prior). Vektörize PL loglik ve
  numerically-stable mask propagation sayesinde 10K yarış 4 dakikada
  fit. Holdout (1.487 yarış): top-1 %35.2 (LGBM %38.6'dan 3.4pp düşük)
  ama Brier −%28 ve logL +0.34 nats — LGBM overconfident, Bayes
  kalibre. Bayes posterior'u **stake sizing + skip-gate** olarak
  kullanılıyor; LGBM hâlâ point-pick rankerı. `ganyan advice`/`/advice`
  Bayes top-1 mean ≥%35 + lo₅ ≥%20 eşiğini geçen yarışları kabul eder;
  Kelly = base × (0.5 + 0.5×conf).

- **2026-04-23 — Cohort filtresi + 11-head ensemble**
  Maiden / dişi / alan ≥13 / Şanlıurfa / Bursa cohort'ları
  bias-yüksek; advice CLI'de varsayılan açık. 14 günlük backtest
  +5,870 TL. Aynı dönemde 11-head LGBM ensemble (rank + value + EV +
  finish-time + 7 spec-class head) 1-head tahminciyle yer
  değiştirdi; AGF snapshot'ları, scraper gate parser fix
  (StartId → SiraId), leakage smoke testleri.

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
