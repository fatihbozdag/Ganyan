# Horse Racing Prediction System

## Overview
A comprehensive horse racing prediction system that uses machine learning and Bayesian analysis to predict race outcomes. The system analyzes various factors including horse performance history, jockey statistics, and race conditions to provide detailed predictions.

## Features
- Machine Learning based predictions
- Bayesian probability analysis
- Combined prediction system
- Web interface for easy data input and analysis
- Real-time prediction updates
- Comprehensive horse statistics tracking

## Project Structure
```
horse_racing/
├── app.py                 # Flask web application
├── race_analyzer.py       # Main race analysis logic
├── bayesian_predictor.py  # Bayesian prediction system
├── templates/            
│   └── index.html        # Web interface template
├── data/
│   ├── processed/        # Processed race data
│   └── raw/              # Raw race data
├── scripts/
│   ├── preprocess_races.py    # Data preprocessing
│   └── standardize_race_data.py # Data standardization
├── analysis/
│   └── race_analysis_patterns.md # Analysis documentation
└── tests/                # Test files
```

## Setup
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python scripts/create_db_from_processed.py
```

## Usage
1. Start the web application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5002
```

3. Enter horse data and view predictions in real-time.

## Data Input Format
- **Horse Name**: Full name of the horse
- **Age**: Horse's age
- **Weight**: Race weight in kg
- **Jockey**: Jockey's name
- **Start Position**: Starting gate position
- **HP (Handikap Puanı)**: Handicap score
- **Last 6**: Results of last 6 races
- **KGS**: Days since last race
- **S20**: Performance in last 20 races
- **EİD**: Best time (format: MM.SS.ms)
- **GNY**: Daily relative race score
- **AGF**: Win probability percentage

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.