#!/usr/bin/env python3
"""
Named Entity Recognition (NER) Script using spaCy

This script extracts named entities from text files using spaCy's pre-trained models.
It implements professional logging practices and follows NLP best practices for
entity extraction and analysis.

"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import json

try:
    import spacy
    from spacy import displacy
except ImportError as e:
    print("Error: spaCy is not installed. Please install it using:")
    print("pip install spacy")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)


class EntityExtractor:
    """
    A professional Named Entity Recognition (NER) class using spaCy.
    
    This class provides comprehensive entity extraction capabilities with
    detailed logging and analysis features commonly used in academic and
    professional NLP applications.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm", log_level: str = "INFO"):
        """
        Initialize the EntityExtractor with specified spaCy model and logging level.
        
        Args:
            model_name (str): Name of the spaCy model to use
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.model_name = model_name
        self.nlp = None
        self.logger = self._setup_logging(log_level)
        self._load_model()
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """
        Set up comprehensive logging configuration.
        
        Args:
            log_level (str): The logging level to use
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger
        logger = logging.getLogger('EntityExtractor')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create file handler for persistent logging
        file_handler = logging.FileHandler('entity_extraction.log', mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_model(self) -> None:
        """
        Load the specified spaCy model with error handling.
        """
        try:
            self.logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            self.logger.info(f"Successfully loaded model: {self.model_name}")
            
            # Log model capabilities
            self.logger.debug(f"Model pipeline components: {self.nlp.pipe_names}")
            self.logger.debug(f"Model language: {self.nlp.lang}")
            
        except OSError as e:
            self.logger.error(f"Failed to load spaCy model '{self.model_name}': {e}")
            self.logger.error("Please install the model using: python -m spacy download en_core_web_sm")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading model: {e}")
            raise
    
    def read_text_file(self, file_path: str) -> str:
        """
        Read text from a file with proper error handling and encoding detection.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: The content of the file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnicodeDecodeError: If the file can't be decoded
        """
        path = Path(file_path)
        
        if not path.exists():
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            self.logger.info(f"Reading file: {file_path}")
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            self.logger.info(f"Successfully read {len(content)} characters from {file_path}")
            self.logger.debug(f"File size: {path.stat().st_size} bytes")
            
            return content
            
        except UnicodeDecodeError as e:
            self.logger.error(f"Unicode decode error reading {file_path}: {e}")
            # Try with different encoding
            try:
                with open(path, 'r', encoding='latin-1') as file:
                    content = file.read()
                self.logger.warning(f"Successfully read file using latin-1 encoding")
                return content
            except Exception as e2:
                self.logger.error(f"Failed to read file with alternative encoding: {e2}")
                raise
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def extract_entities(self, text: str) -> Tuple[spacy.tokens.Doc, Dict]:
        """
        Extract named entities from text using spaCy NER.
        
        Args:
            text (str): Input text to process
            
        Returns:
            Tuple[spacy.tokens.Doc, Dict]: Processed spaCy doc and entity statistics
        """
        if not text.strip():
            self.logger.warning("Empty or whitespace-only text provided")
            return None, {}
        
        try:
            self.logger.info("Starting entity extraction process")
            self.logger.debug(f"Processing text of length: {len(text)} characters")
            
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract entities and create statistics
            entities = list(doc.ents)
            self.logger.info(f"Found {len(entities)} named entities")
            
            # Create comprehensive statistics
            entity_stats = self._analyze_entities(entities)
            
            # Log entity type distribution
            for ent_type, count in entity_stats['type_counts'].items():
                self.logger.debug(f"{ent_type}: {count} entities")
            
            return doc, entity_stats
            
        except Exception as e:
            self.logger.error(f"Error during entity extraction: {e}")
            raise
    
    def _analyze_entities(self, entities: List[spacy.tokens.Span]) -> Dict:
        """
        Perform comprehensive analysis of extracted entities.
        
        Args:
            entities (List[spacy.tokens.Span]): List of entity spans
            
        Returns:
            Dict: Comprehensive entity statistics and analysis
        """
        self.logger.debug("Analyzing entity statistics")
        
        # Count entities by type
        type_counts = Counter(ent.label_ for ent in entities)
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for ent in entities:
            entities_by_type[ent.label_].append({
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            })
        
        # Calculate confidence scores if available
        confidence_scores = []
        for ent in entities:
            if hasattr(ent, 'kb_id_') and ent.kb_id_:
                confidence_scores.append(float(ent.kb_id_))
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
        
        stats = {
            'total_entities': len(entities),
            'unique_types': len(type_counts),
            'type_counts': dict(type_counts),
            'entities_by_type': dict(entities_by_type),
            'average_confidence': avg_confidence,
            'entity_density': len(entities) / len(entities[0].doc.text) if entities else 0
        }
        
        self.logger.debug(f"Analysis complete: {stats['total_entities']} entities, {stats['unique_types']} types")
        
        return stats
    
    def print_entities(self, doc: spacy.tokens.Doc, entity_stats: Dict, 
                      show_details: bool = True) -> None:
        """
        Print extracted entities in a formatted, professional manner.
        
        Args:
            doc (spacy.tokens.Doc): Processed spaCy document
            entity_stats (Dict): Entity statistics dictionary
            show_details (bool): Whether to show detailed entity information
        """
        print("\n" + "="*80)
        print("NAMED ENTITY RECOGNITION RESULTS")
        print("="*80)
        
        print(f"\nSUMMARY:")
        print(f"Total Entities Found: {entity_stats['total_entities']}")
        print(f"Unique Entity Types: {entity_stats['unique_types']}")
        print(f"Entity Density: {entity_stats['entity_density']:.4f} entities per character")
        
        if entity_stats['average_confidence']:
            print(f"Average Confidence: {entity_stats['average_confidence']:.3f}")
        
        print(f"\nENTITY TYPE DISTRIBUTION:")
        print("-" * 40)
        for ent_type, count in sorted(entity_stats['type_counts'].items(), 
                                     key=lambda x: x[1], reverse=True):
            description = spacy.explain(ent_type) or "Unknown"
            print(f"{ent_type:12} | {count:3d} | {description}")
        
        if show_details:
            print(f"\nDETAILED ENTITY LIST:")
            print("-" * 60)
            for ent_type, entities in entity_stats['entities_by_type'].items():
                print(f"\n{ent_type} ({spacy.explain(ent_type) or 'Unknown'}):")
                for ent in entities:
                    print(f"  • {ent['text']} (chars {ent['start']}-{ent['end']})")
        
        print("\n" + "="*80)
    
    def save_results(self, entity_stats: Dict, output_file: str = "entity_results.json") -> None:
        """
        Save entity extraction results to a JSON file.
        
        Args:
            entity_stats (Dict): Entity statistics to save
            output_file (str): Output file path
        """
        try:
            self.logger.info(f"Saving results to: {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entity_stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results successfully saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def generate_html_visualization(self, doc: spacy.tokens.Doc, 
                                  output_file: str = "entities_visualization.html") -> None:
        """
        Generate an HTML visualization of the entities using spaCy's displacy.
        
        Args:
            doc (spacy.tokens.Doc): Processed spaCy document
            output_file (str): Output HTML file path
        """
        try:
            self.logger.info(f"Generating HTML visualization: {output_file}")
            
            html = displacy.render(doc, style="ent", page=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            
            self.logger.info(f"HTML visualization saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating HTML visualization: {e}")
            raise


def main():
    """
    Main function with comprehensive command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Professional Named Entity Recognition using spaCy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python entity_extractor.py                           # Use default article.txt
  python entity_extractor.py -f my_text.txt          # Use custom file
  python entity_extractor.py -v DEBUG                # Enable debug logging
  python entity_extractor.py -m en_core_web_lg       # Use large model
  python entity_extractor.py --save-json results.json # Save to custom JSON
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        default='article.txt',
        help='Input text file path (default: article.txt)'
    )
    
    parser.add_argument(
        '-m', '--model',
        default='en_core_web_sm',
        help='spaCy model to use (default: en_core_web_sm)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging verbosity level (default: INFO)'
    )
    
    parser.add_argument(
        '--save-json',
        help='Save results to JSON file (optional)'
    )
    
    parser.add_argument(
        '--save-html',
        help='Save HTML visualization (optional)'
    )
    
    parser.add_argument(
        '--no-details',
        action='store_true',
        help='Hide detailed entity list in output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize entity extractor
        extractor = EntityExtractor(model_name=args.model, log_level=args.verbose)
        
        # Read input file
        text = extractor.read_text_file(args.file)
        
        # Extract entities
        doc, entity_stats = extractor.extract_entities(text)
        
        if doc and entity_stats:
            # Print results
            extractor.print_entities(doc, entity_stats, show_details=not args.no_details)
            
            # Save JSON results if requested
            if args.save_json:
                extractor.save_results(entity_stats, args.save_json)
            
            # Save HTML visualization if requested
            if args.save_html:
                extractor.generate_html_visualization(doc, args.save_html)
        
        extractor.logger.info("Entity extraction completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 