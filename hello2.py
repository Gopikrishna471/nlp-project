import re
from typing import Dict, List
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np

class NewsContentExtractor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stopwords = set(stopwords.words('english'))
        
        # Common patterns to filter out
        self.noise_patterns = [
            r'Copyright ©.*$',
            r'https?://\S+',
            r'Follow us on.*$',
            r'Click here.*$',
            r'Subscribe.*$',
            r'Advertisement.*$',
            r'About the Author.*$',
            r'Share this:.*$',
            r'Tags:.*$',
            r'Related Articles.*$',
            r'Comments.*$'
        ]
        
        # Navigation and footer terms to filter out
        self.noise_terms = {
            'trending', 'privacy policy', 'terms of use', 'contact us',
            'follow us', 'newsletter', 'subscribe', 'advertisement',
            'copyright', 'all rights reserved', 'social media',
            'related stories', 'popular stories', 'most read'
        }

    def clean_text(self, text: str) -> str:
        """Remove noise patterns and unnecessary whitespace."""
        # Remove noise patterns
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def is_content_sentence(self, sentence: str) -> bool:
        """Check if a sentence is likely to be main content."""
        # Convert to lowercase for comparison
        sentence_lower = sentence.lower()
        
        # Check if sentence contains noise terms
        if any(term in sentence_lower for term in self.noise_terms):
            return False
        
        # Check sentence length (too short sentences are likely noise)
        if len(sentence.split()) < 4:
            return False
        
        # Check if sentence is mostly special characters or numbers
        alpha_ratio = sum(c.isalpha() for c in sentence) / len(sentence) if sentence else 0
        if alpha_ratio < 0.5:
            return False
        
        return True

    def extract_content(self, text: str) -> str:
        """Extract main content from the text."""
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Split into sentences
        sentences = sent_tokenize(cleaned_text)
        
        # Filter content sentences
        content_sentences = [s for s in sentences if self.is_content_sentence(s)]
        
        # Join sentences back together
        main_content = ' '.join(content_sentences)
        
        return main_content

    def get_article_metadata(self, text: str) -> Dict:
        """Extract basic metadata from the article."""
        lines = text.split('\n')
        metadata = {
            'title': '',
            'date': '',
            'author': ''
        }
        
        # Try to find title (usually one of the first few non-empty lines)
        for line in lines[:10]:
            if len(line.strip()) > 20 and '|' not in line and ':' not in line:
                metadata['title'] = line.strip()
                break
        
        # Look for date patterns
        date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        dates = re.findall(date_pattern, text)
        if dates:
            metadata['date'] = dates[0]
        
        # Look for author patterns
        author_patterns = [
            r'By\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'Author:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        for pattern in author_patterns:
            match = re.search(pattern, text)
            if match:
                metadata['author'] = match.group(1)
                break
        
        return metadata

def main(text: str) -> Dict:
    """Main function to process article text."""
    extractor = NewsContentExtractor()
    
    # Extract metadata
    metadata = extractor.get_article_metadata(text)
    
    # Extract main content
    content = extractor.extract_content(text)
    
    return {
        'metadata': metadata,
        'content': content
    }


# Example usage
text = """[Your article text here]"""
result = main(text)

print("Title:", result['metadata']['title'])
print("Date:", result['metadata']['date'])
print("Author:", result['metadata']['author'])
print("\nMain Content:")
print(result['content'])