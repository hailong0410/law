"""
Chunking strategies for documents in RAG system.
"""

from typing import List, Dict, Any, Callable
from enum import Enum
import re
from abc import ABC, abstractmethod


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, max_length: int) -> List[str]:
        """
        Chunk text according to strategy.
        
        Args:
            text: Text to chunk
            max_length: Maximum length of each chunk
        
        Returns:
            List of chunks
        """
        pass


class CharacterChunking(ChunkingStrategy):
    """Split text by characters."""
    
    def chunk(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text by characters with overlap.
        
        Args:
            text: Text to chunk
            max_length: Maximum chunk size
        
        Returns:
            List of character chunks
        """
        if not text or max_length <= 0:
            return []
        
        chunks = []
        overlap = max(max_length // 10, 50)  # 10% overlap
        
        start = 0
        while start < len(text):
            end = min(start + max_length, len(text))
            chunk = text[start:end]
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks


class LineChunking(ChunkingStrategy):
    """Split text by lines."""
    
    def chunk(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text by lines, respecting max_length.
        
        Args:
            text: Text to chunk
            max_length: Maximum chunk size in characters
        
        Returns:
            List of line-based chunks
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            # If adding this line exceeds max_length
            if current_length + line_length > max_length and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # If single line exceeds max_length, split it recursively
            if line_length > max_length:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long line using character chunking
                char_chunker = CharacterChunking()
                for sub_chunk in char_chunker.chunk(line, max_length):
                    chunks.append(sub_chunk)
            else:
                current_chunk.append(line)
                current_length += line_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]


class SentenceChunking(ChunkingStrategy):
    """Split text by sentences."""
    
    def chunk(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text by sentences, respecting max_length.
        
        Args:
            text: Text to chunk
            max_length: Maximum chunk size in characters
        
        Returns:
            List of sentence-based chunks
        """
        # Split by common sentence delimiters
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 1  # +1 for space
            
            # If adding this sentence exceeds max_length
            if current_length + sentence_length > max_length and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # If single sentence exceeds max_length, split by character
            if sentence_length > max_length:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence using character chunking
                char_chunker = CharacterChunking()
                for sub_chunk in char_chunker.chunk(sentence, max_length):
                    chunks.append(sub_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]


class MarkdownChunking(ChunkingStrategy):
    """Split markdown text by headers."""
    
    def chunk(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split markdown text by headers (# ## ### etc), respecting max_length.
        
        Args:
            text: Markdown text to chunk
            max_length: Maximum chunk size in characters
        
        Returns:
            List of markdown-based chunks
        """
        # Split by markdown headers
        # Pattern: # or ## or ### etc followed by text
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            is_header = re.match(header_pattern, line)
            line_length = len(line) + 1  # +1 for newline
            
            # If it's a header and we have accumulated content
            if is_header and current_chunk and current_length > 0:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # If adding this line exceeds max_length
            if current_length + line_length > max_length and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # If line exceeds max_length, split it
            if line_length > max_length:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Use character chunking for very long lines
                char_chunker = CharacterChunking()
                for sub_chunk in char_chunker.chunk(line, max_length):
                    chunks.append(sub_chunk)
            else:
                current_chunk.append(line)
                current_length += line_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]


class ParagraphChunking(ChunkingStrategy):
    """Split text by paragraphs."""
    
    def chunk(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text by paragraphs (separated by blank lines), respecting max_length.
        
        Args:
            text: Text to chunk
            max_length: Maximum chunk size in characters
        
        Returns:
            List of paragraph-based chunks
        """
        # Split by one or more blank lines
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para) + 1  # +1 for space between paragraphs
            
            # If adding this paragraph exceeds max_length
            if current_length + para_length > max_length and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # If single paragraph exceeds max_length
            if para_length > max_length:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split paragraph by sentences
                sentence_chunker = SentenceChunking()
                for sub_chunk in sentence_chunker.chunk(para, max_length):
                    chunks.append(sub_chunk)
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]


class ChunkingFactory:
    """Factory for creating chunking strategies."""
    
    _strategies: Dict[str, ChunkingStrategy] = {
        'character': CharacterChunking(),
        'line': LineChunking(),
        'sentence': SentenceChunking(),
        'markdown': MarkdownChunking(),
        'paragraph': ParagraphChunking(),
    }
    
    @classmethod
    def get_strategy(cls, strategy_name: str) -> ChunkingStrategy:
        """
        Get a chunking strategy by name.
        
        Args:
            strategy_name: Name of the strategy
        
        Returns:
            ChunkingStrategy instance
        
        Raises:
            ValueError: If strategy not found
        """
        if strategy_name not in cls._strategies:
            raise ValueError(
                f"Unknown chunking strategy: {strategy_name}. "
                f"Available: {list(cls._strategies.keys())}"
            )
        return cls._strategies[strategy_name]
    
    @classmethod
    def register_strategy(cls, name: str, strategy: ChunkingStrategy) -> None:
        """
        Register a custom chunking strategy.
        
        Args:
            name: Name for the strategy
            strategy: ChunkingStrategy instance
        """
        cls._strategies[name] = strategy
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available chunking strategies."""
        return list(cls._strategies.keys())


class ChunkedDocument:
    """Represents a document with its chunks and metadata."""
    
    def __init__(
        self,
        document_id: str,
        content: str,
        chunks: List[str],
        metadata: Dict[str, Any],
        chunking_strategy: str
    ):
        self.document_id = document_id
        self.content = content
        self.chunks = chunks
        self.metadata = metadata
        self.chunking_strategy = chunking_strategy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "content_length": len(self.content),
            "num_chunks": len(self.chunks),
            "chunks": self.chunks,
            "metadata": self.metadata,
            "chunking_strategy": self.chunking_strategy
        }
