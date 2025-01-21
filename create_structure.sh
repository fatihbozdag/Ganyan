mkdir horse_racing_predictor
cd horse_racing_predictor

# Create main project directories
mkdir -p src/scrapers
mkdir -p src/models
mkdir -p src/utils
mkdir -p data/raw
mkdir -p data/processed
mkdir -p tests
mkdir -p notebooks

# Create initial files
touch README.md
touch requirements.txt
touch src/__init__.py
touch src/scrapers/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py

# Move all files from src/ to root if they exist
mv src/scrapers scrapers
mv src/scripts scripts
mv src/utils utils
rm -rf src  # Remove the empty src directory

# Make sure we have __init__.py files in each directory
touch __init__.py
touch scrapers/__init__.py
touch scripts/__init__.py
touch utils/__init__.py

# Create setup.py
touch setup.py 