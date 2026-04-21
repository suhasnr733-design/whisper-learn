"""
Flashcard generation from lecture content
"""
from typing import List, Dict
import random

class FlashcardGenerator:
    def __init__(self, llm=None):
        self.llm = llm
        
    def generate_from_text(self, text: str, num_cards: int = 10) -> List[Dict]:
        """Generate flashcards from text"""
        
        # Extract key sentences
        sentences = self._extract_key_sentences(text, num_cards * 2)
        
        flashcards = []
        for sentence in sentences[:num_cards]:
            # Create Q&A pair from sentence
            flashcard = self._create_flashcard(sentence)
            flashcards.append(flashcard)
        
        return flashcards
    
    def _extract_key_sentences(self, text: str, num_sentences: int) -> List[str]:
        """Extract important sentences using simple heuristics"""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        
        # Score sentences by length and keyword presence
        keywords = ['important', 'key', 'definition', 'example', 'concept', 
                   'therefore', 'thus', 'consequently', 'significant']
        
        scores = []
        for sent in sentences:
            score = len(sent.split())  # Length score
            for kw in keywords:
                if kw in sent.lower():
                    score += 10
            scores.append(score)
        
        # Get top sentences
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        
        top_indices = [idx for idx, _ in indexed[:num_sentences]]
        top_indices.sort()  # Keep original order
        
        return [sentences[i] for i in top_indices]
    
    def _create_flashcard(self, sentence: str) -> Dict:
        """Create Q&A flashcard from sentence"""
        
        # Try to identify the key concept
        words = sentence.split()
        
        # Find potential key terms (capitalized or long words)
        key_terms = [w for w in words if w[0].isupper() or len(w) > 6]
        
        if key_terms:
            question = f"What is {key_terms[0]}?"
        else:
            # Generic question
            question = f"Explain: {sentence[:50]}..."
        
        return {
            'question': question,
            'answer': sentence,
            'source_sentence': sentence
        }
    
    def generate_multiple_choice(self, text: str, num_questions: int = 5) -> List[Dict]:
        """Generate multiple choice questions"""
        flashcards = self.generate_from_text(text, num_questions)
        
        mc_questions = []
        for card in flashcards:
            # Generate distractors (simplified)
            answer = card['answer']
            words = answer.split()
            
            # Create simple distractors by modifying key words
            distractor1 = answer.replace(words[min(3, len(words)-1)], "something else")
            distractor2 = answer.replace(words[min(2, len(words)-1)], "the opposite")
            distractor3 = "None of the above"
            
            mc_questions.append({
                'question': card['question'],
                'correct_answer': answer,
                'distractors': [distractor1, distractor2, distractor3],
                'all_answers': [answer, distractor1, distractor2, distractor3]
            })
        
        return mc_questions
    
    def export_anki(self, flashcards: List[Dict], output_file: str):
        """Export to Anki deck format"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Question', 'Answer'])
            
            for card in flashcards:
                writer.writerow([card['question'], card['answer']])
        
        return output_file