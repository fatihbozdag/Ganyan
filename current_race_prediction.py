from race_analyzer import RaceAnalyzer

# Initialize analyzer
analyzer = RaceAnalyzer(db_path='races_new.db')

# Race information
race_info = {
    'track': 'Adana',
    'time': '21:30',
    'distance': '2100',
    'surface': 'Kum',
    'race_type': 'Handikap-16',
    'horse_type': '4+ Araplar',
    'race_weight': '54.5-61.5'
}

# Horse entries
entries = [
    {
        "number": "1",
        "name": "CEBEALP",
        "origin": "DB",
        "age": "6y a a",
        "sire": "ÖZGÜNHAN",
        "dam": "SÜHUNET",
        "weight": "60",
        "jockey": "M.KIYAK",
        "owner_trainer": "BEŞİR KAYA/A.DİNÇ",
        "start_pos": "5",
        "hp": "72",
        "last_6": "620008",
        "kgs": "21",
        "s20": "13",
        "eid": "2.30.25",
        "agf": "1.31"
    },
    {
        "number": "2",
        "name": "BEYAZ ADAM",
        "origin": "DB SKG SK",
        "age": "10y a a",
        "sire": "GOBAKBEY",
        "dam": "İPONONİ",
        "weight": "60.5",
        "jockey": "K.DEMİREL",
        "owner_trainer": "SELİM ÖZEKİCİ/EMİN KARATAŞ",
        "start_pos": "2",
        "hp": "71",
        "last_6": "326375",
        "kgs": "12",
        "s20": "15",
        "eid": "2.31.11",
        "agf": "4.71"
    },
    {
        "number": "3",
        "name": "BEDOBEY",
        "origin": "KG K DB",
        "age": "5y a a",
        "sire": "SAKARBAŞI",
        "dam": "AYYALAZ",
        "weight": "60",
        "jockey": "F.S.M.SANSAR",
        "owner_trainer": "ASUMAN YENİCE/M.ATAL",
        "start_pos": "13",
        "hp": "70",
        "last_6": "389153",
        "kgs": "14",
        "s20": "18",
        "eid": "2.34.88",
        "agf": "10.84"
    },
    {
        "number": "4",
        "name": "ASLAN YAĞIZ",
        "origin": "KG K DS",
        "age": "5y k a",
        "sire": "ANKA",
        "dam": "SARAYGÜLÜ",
        "weight": "61.5",
        "jockey": "E.ÇANKAYA",
        "owner_trainer": "MAHMUT TURANOĞLU/A.ÇIKMAN",
        "start_pos": "16",
        "hp_score": "69",
        "last_6": "565135",
        "kgs_score": "14",
        "s20_score": "18",
        "eid_score": "2.35.85"
    },
    {
        "number": "5",
        "name": "SARRAFTAY",
        "origin": "KG K DB",
        "age": "7y d a",
        "sire": "SARRAF",
        "dam": "ÇIKMAZNUR",
        "weight": "59",
        "jockey": "Y.ÇELİKBAŞ",
        "owner_trainer": "BUBO ERCAN/AH.KORKMAZ",
        "start_pos": "15",
        "hp_score": "68",
        "last_6": "466395",
        "kgs_score": "13",
        "s20_score": "14",
        "eid_score": "2.32.51"
    },
    {
        "number": "6",
        "name": "ÖZKEHRİBAR",
        "origin": "DB SK",
        "age": "5y a k",
        "sire": "ÖZGÜNHAN",
        "dam": "KEHRİBAR",
        "weight": "59.5",
        "jockey": "M.AKYAVUZ",
        "owner_trainer": "MEHMET KARATOPRAK/A.ÇİFTÇİ",
        "start_pos": "4",
        "hp_score": "65",
        "last_6": "154246",
        "kgs_score": "17",
        "s20_score": "16",
        "eid_score": "2.33.68"
    },
    {
        "number": "7",
        "name": "MİRİALEM",
        "origin": "KG DB SK",
        "age": "6y d a",
        "sire": "SARRAF",
        "dam": "DAĞÇİÇEĞİ",
        "weight": "59",
        "jockey": "N.DEMİR",
        "owner_trainer": "HAYDAR YARDIMCI/MÜS.YARDIMCI",
        "start_pos": "6",
        "hp_score": "64",
        "last_6": "639320",
        "kgs_score": "8",
        "s20_score": "14",
        "eid_score": "2.33.16"
    },
    {
        "number": "8",
        "name": "AYHANBEYİM",
        "origin": "KG K DB",
        "age": "5y a a",
        "sire": "SONALP",
        "dam": "NEZAHAT",
        "weight": "56",
        "jockey": "İSM.YILDIRIM",
        "owner_trainer": "ADİL ALTIN/HAS.YILDIRIM",
        "start_pos": "12",
        "hp_score": "62",
        "last_6": "134650",
        "kgs_score": "8",
        "s20_score": "15",
        "eid_score": "2.33.32"
    },
    {
        "number": "9",
        "name": "MUŞFIRTINASI",
        "origin": "KG K",
        "age": "5y k a",
        "sire": "DİZDAR BEY",
        "dam": "ZELYURT GÜLÜ",
        "weight": "57.5",
        "jockey": "H.ÇİZİK",
        "owner_trainer": "GÜNAYDIN SAÇAN/S.BECENE",
        "start_pos": "3",
        "hp": "61",
        "last_6": "146381",
        "kgs": "11",
        "s20": "20",
        "eid": "2.43.24",
        "agf": "8.36"
    },
    {
        "number": "10",
        "name": "RAJBERA",
        "origin": "KG K DS",
        "age": "5y k a",
        "sire": "SERHANTAY",
        "dam": "KAMERAY",
        "weight": "57.5",
        "jockey": "A.ÇELİK",
        "owner_trainer": "NECDET YILDIZ/Y.TUNCAY",
        "start_pos": "18",
        "hp": "61",
        "last_6": "315117",
        "kgs": "8",
        "s20": "19",
        "eid": "2.35.37",
        "agf": "16.37"
    },
    {
        "number": "11",
        "name": "ATEŞ AYAZ",
        "origin": "K DB",
        "age": "5y a a",
        "sire": "DERHADIR",
        "dam": "KANDEZİ",
        "weight": "55",
        "jockey": "F.ÇETİNBAŞ",
        "owner_trainer": "GİZEM ÇAKMAKOĞLU/Y.TUNCAY",
        "start_pos": "14",
        "hp_score": "60",
        "last_6": "153733",
        "kgs_score": "8",
        "s20_score": "16",
        "eid_score": "2.35.11"
    },
    {
        "number": "12",
        "name": "MADENCİ",
        "origin": "KG DB SK",
        "age": "5y k a",
        "sire": "ENDEREFE",
        "dam": "YILDIZ TÜRE",
        "weight": "55",
        "jockey": "M.N.SUNKAR",
        "owner_trainer": "ABDULLAH ÇERMAN/S.BECENE",
        "start_pos": "10",
        "hp": "60",
        "last_6": "267328",
        "kgs": "34",
        "s20": "15",
        "eid": "2.34.98",
        "agf": "3.67"
    },
    {
        "number": "13",
        "name": "DOBAR",
        "origin": "KG DB DS",
        "age": "7y a a",
        "sire": "BAYHAN",
        "dam": "DÜRDANE",
        "weight": "55.5",
        "jockey": "E.SİNCAN",
        "owner_trainer": "ÖMER POLAT/S.POLAT",
        "start_pos": "17",
        "hp_score": "57",
        "last_6": "381570",
        "kgs_score": "8",
        "s20_score": "17",
        "eid_score": "2.31.76"
    },
    {
        "number": "14",
        "name": "ÇAĞLAYAN",
        "origin": "KG K DB",
        "age": "5y k a",
        "sire": "TAMERİNOĞLU",
        "dam": "SEÇİLAY",
        "weight": "55.5",
        "jockey": "E.ÇİZİK",
        "owner_trainer": "MUSTAFA TÜZÜN/Ş.AYDEMİR",
        "start_pos": "11",
        "hp_score": "57",
        "last_6": "224119",
        "kgs_score": "12",
        "s20_score": "19",
        "eid_score": "2.34.01"
    },
    {
        "number": "15",
        "name": "KIRKÇERİ",
        "origin": "KG K DB",
        "age": "6y d a",
        "sire": "ULUER",
        "dam": "KIRIMKIZ",
        "weight": "55",
        "jockey": "F.ÇETİN",
        "owner_trainer": "CELAL AKIL/S.BECENE",
        "start_pos": "9",
        "hp_score": "56",
        "last_6": "462259",
        "kgs_score": "22",
        "s20_score": "17",
        "eid_score": "2.36.16"
    },
    {
        "number": "16",
        "name": "PALA SALİH",
        "origin": "",
        "age": "10y k a",
        "sire": "KIRIMHAN",
        "dam": "ÖMÜRÜM",
        "weight": "55",
        "jockey": "MÜS.ÇELİK",
        "owner_trainer": "YAKUP KAYA/Y.TUNCAY",
        "start_pos": "1",
        "hp_score": "56",
        "last_6": "062228",
        "kgs_score": "8",
        "s20_score": "18",
        "eid_score": "2.32.56"
    },
    {
        "number": "17",
        "name": "BUĞRAHAN",
        "origin": "KG",
        "age": "10y k a",
        "sire": "ARASLI",
        "dam": "NAZPERVER",
        "weight": "55",
        "jockey": "M.S.ÇELİK",
        "owner_trainer": "HÜSEYİN ATAKAYA/H.ATAKAYA",
        "start_pos": "7",
        "hp_score": "56",
        "last_6": "184755",
        "kgs_score": "11",
        "s20_score": "15",
        "eid_score": "2.39.63"
    },
    {
        "number": "18",
        "name": "ENFES TAMER",
        "origin": "DB",
        "age": "5y k a",
        "sire": "TAMERİNOĞLU",
        "dam": "ENFES",
        "weight": "54.5",
        "jockey": "V.DEMİR",
        "owner_trainer": "ÖMER FELHAN/A.AKBABA",
        "start_pos": "8",
        "hp_score": "55",
        "last_6": "371228",
        "kgs_score": "11",
        "s20_score": "18",
        "eid_score": "2.52.66"
    }
]

# Run analysis
print("\nAnalyzing race...")
predictions = analyzer.analyze_race(race_info, entries)

# Sort predictions by total score
sorted_predictions = sorted(predictions, key=lambda x: x['total_score'], reverse=True)

# Print results
print("\nPredictions (in order of likelihood):")
print("-" * 50)

for pred in sorted_predictions:
    print(f"\n{pred['horse_name']}:")
    print(f"Total Score: {pred['total_score']:.2f}")
    print(f"Base Score: {pred['base_score']:.2f}")
    print("Detailed Scores:")
    for key, value in pred['detailed_scores'].items():
        print(f"  {key}: {value:.2f}")
    print("Historical Performance:")
    for key, value in pred['historical_stats'].items():
        print(f"  {key}: {value}") 