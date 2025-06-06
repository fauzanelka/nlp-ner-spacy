# Named Entity Recognition (NER) with spaCy

A professional-grade Python script for Named Entity Recognition using spaCy, designed for both academic research and professional NLP applications.

[![asciicast](https://asciinema.org/a/OM5yRnwmnzWos1FTV3yhvlOBy.svg)](https://asciinema.org/a/OM5yRnwmnzWos1FTV3yhvlOBy)

## Overview

This project implements a comprehensive Named Entity Recognition system that extracts and analyzes named entities from text files. It features extensive logging, statistical analysis, and visualization capabilities commonly required in academic and professional NLP workflows.

## Features

- **Professional Logging**: Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL) with both console and file output
- **Robust Entity Extraction**: Uses spaCy's state-of-the-art NER models
- **Comprehensive Analysis**: Detailed statistics including entity counts, types, density, and confidence scores
- **Multiple Output Formats**: Console display, JSON export, and HTML visualization
- **Error Handling**: Graceful handling of file encoding issues and missing dependencies
- **Command-Line Interface**: Flexible CLI with multiple options and configurations
- **Academic Standards**: Follows best practices for reproducible research

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model** (if not automatically installed):
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Alternative Installation Methods

For different spaCy models:

```bash
# Medium model (more accurate, larger)
python -m spacy download en_core_web_md

# Large model (most accurate, largest)
python -m spacy download en_core_web_lg
```

## Usage

### Basic Usage

Process the default `article.txt` file:
```bash
python entity_extractor.py
```

### Advanced Usage

```bash
# Use a custom input file
python entity_extractor.py -f my_document.txt

# Enable debug logging
python entity_extractor.py -v DEBUG

# Use a larger, more accurate model
python entity_extractor.py -m en_core_web_lg

# Save results to JSON
python entity_extractor.py --save-json my_results.json

# Generate HTML visualization
python entity_extractor.py --save-html entities.html

# Hide detailed entity list
python entity_extractor.py --no-details

# Combine multiple options
python entity_extractor.py -f research_paper.txt -m en_core_web_lg -v DEBUG --save-json results.json --save-html viz.html
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --file` | Input text file path | `article.txt` |
| `-m, --model` | spaCy model to use | `en_core_web_sm` |
| `-v, --verbose` | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) | `INFO` |
| `--save-json` | Save results to JSON file | None |
| `--save-html` | Save HTML visualization | None |
| `--no-details` | Hide detailed entity list | False |

## Output

The script provides comprehensive output including:

1. **Summary Statistics**:
   - Total entities found
   - Number of unique entity types
   - Entity density (entities per character)
   - Average confidence scores (when available)

2. **Entity Type Distribution**:
   - Count of each entity type (PERSON, ORG, GPE, etc.)
   - Human-readable descriptions of entity types

3. **Detailed Entity List**:
   - Individual entities grouped by type
   - Character positions in source text
   - Entity labels and descriptions

## Entity Types

The script recognizes standard spaCy entity types:

- **PERSON**: People, including fictional
- **NORP**: Nationalities or religious or political groups
- **FAC**: Buildings, airports, highways, bridges, etc.
- **ORG**: Companies, agencies, institutions, etc.
- **GPE**: Countries, cities, states
- **LOC**: Non-GPE locations, mountain ranges, bodies of water
- **PRODUCT**: Objects, vehicles, foods, etc. (not services)
- **EVENT**: Named hurricanes, battles, wars, sports events, etc.
- **WORK_OF_ART**: Titles of books, songs, etc.
- **LAW**: Named documents made into laws
- **LANGUAGE**: Any named language
- **DATE**: Absolute or relative dates or periods
- **TIME**: Times smaller than a day
- **PERCENT**: Percentage, including "%"
- **MONEY**: Monetary values, including unit
- **QUANTITY**: Measurements, as of weight or distance
- **ORDINAL**: "first", "second", etc.
- **CARDINAL**: Numerals that do not fall under another type

## Academic Context

### Why spaCy?

spaCy is chosen for this implementation because:

1. **State-of-the-Art Performance**: Uses transformer-based models for high accuracy
2. **Industrial Strength**: Designed for production use with speed optimizations
3. **Academic Adoption**: Widely used in academic research and cited in papers
4. **Reproducibility**: Consistent results across different environments
5. **Extensive Documentation**: Well-documented with clear model provenance

### Research Applications

This tool is suitable for:

- **Corpus Linguistics**: Large-scale entity analysis in text corpora
- **Digital Humanities**: Entity extraction from historical documents
- **Information Extraction**: Automated knowledge base construction
- **Social Media Analysis**: Entity recognition in social media texts
- **News Analytics**: Named entity tracking in news articles
- **Legal Document Processing**: Entity extraction from legal texts

### Best Practices Implemented

1. **Logging**: Comprehensive logging for research reproducibility
2. **Error Handling**: Robust error handling for diverse text inputs
3. **Statistical Analysis**: Detailed metrics for research reporting
4. **Output Formats**: Multiple output formats for different analysis needs
5. **Model Flexibility**: Support for different spaCy models
6. **Documentation**: Extensive code documentation and type hints

## File Structure

```
nlp-personal-assignment-1/
├── entity_extractor.py      # Main script
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── article.txt             # Sample input file
├── entity_extraction.log   # Log file (created when run)
├── entity_results.json     # Results (when --save-json used)
└── entities_visualization.html  # Visualization (when --save-html used)
```

## Troubleshooting

### Common Issues

1. **Model Not Found Error**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Unicode Encoding Issues**:
   The script automatically tries alternative encodings (latin-1) if UTF-8 fails.

3. **Memory Issues with Large Files**:
   Consider using the smaller `en_core_web_sm` model for very large texts.

### Performance Considerations

- **en_core_web_sm**: Fast, good for most applications (~15MB)
- **en_core_web_md**: More accurate with word vectors (~50MB)
- **en_core_web_lg**: Most accurate, largest model (~750MB)

## Contributing

This is an academic assignment project. For improvements or suggestions:

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include type hints
4. Add appropriate logging
5. Ensure backward compatibility

## License

This project is created for academic purposes. Please cite appropriately if used in research.

## References

- Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.
- spaCy Documentation: https://spacy.io/
- Named Entity Recognition: https://en.wikipedia.org/wiki/Named-entity_recognition 