"""
Question answering system using retrieved context
"""
from typing import Dict, List, Optional

class QASystem:
    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = llm
        self.conversation_history = []
        
    def answer(self, question: str, context_docs: Optional[List[str]] = None) -> Dict:
        """Answer question using retrieved context"""
        
        # Retrieve relevant documents if not provided
        if context_docs is None:
            retrieved = self.retriever.retrieve(question, top_k=3)
            context_docs = [r['text'] for r in retrieved]
            scores = [r['similarity'] for r in retrieved]
        else:
            scores = [1.0] * len(context_docs)
        
        # Combine context
        context = '\n\n---\n\n'.join(context_docs)
        
        # Generate answer using LLM if available
        if self.llm:
            answer = self.llm.generate_answer(question, context)
        else:
            # Simple baseline: return most relevant chunk
            answer = context_docs[0] if context_docs else "No relevant context found"
        
        # Store in history
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'context': context_docs,
            'scores': scores
        })
        
        return {
            'question': question,
            'answer': answer,
            'sources': context_docs,
            'confidence': max(scores) if scores else 0,
            'history_length': len(self.conversation_history)
        }
    
    def answer_with_history(self, question: str, use_history: bool = True) -> Dict:
        """Answer with conversation history context"""
        if use_history and self.conversation_history:
            # Add previous Q&A to context
            history_text = '\n'.join([
                f"Previous Q: {h['question']}\nPrevious A: {h['answer']}"
                for h in self.conversation_history[-3:]  # Last 3 exchanges
            ])
            enhanced_question = f"{history_text}\n\nCurrent Q: {question}"
            return self.answer(enhanced_question)
        
        return self.answer(question)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_follow_up_suggestions(self, last_question: str, last_answer: str) -> List[str]:
        """Generate follow-up question suggestions"""
        # Simple keyword-based suggestions
        keywords = ['what', 'why', 'how', 'when', 'where', 'which']
        
        suggestions = []
        for kw in keywords:
            if kw not in last_question.lower():
                suggestions.append(f"{kw} does this relate to...")
        
        return suggestions[:3]