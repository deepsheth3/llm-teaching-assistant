"""
Lesson Generation Service

Converts research paper sections into beginner-friendly lessons.
"""

import time
import asyncio
from typing import Optional, AsyncGenerator
from openai import OpenAI, AsyncOpenAI

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import LessonGenerationError
from models.paper import ParsedPaper, PaperSection
from models.lesson import (
    LessonFragment,
    FullLesson,
    LessonRequest,
    LessonDifficulty,
    StreamingLessonChunk,
)
from services.cache_service import get_cache_service

logger = get_logger(__name__)


class LessonService:
    """
    Service for generating lessons from paper sections.
    
    Features:
    - Beginner-friendly explanations
    - Step-by-step math breakdowns
    - Smooth section transitions
    - Streaming support
    - Caching
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.cache = get_cache_service()
        
        logger.info(f"Lesson service initialized: model={self.settings.lesson_model}")
    
    def generate_lesson(
        self,
        paper: ParsedPaper,
        request: LessonRequest
    ) -> FullLesson:
        """
        Generate a full lesson from a parsed paper.
        
        Args:
            paper: Parsed paper with sections
            request: Lesson generation request
            
        Returns:
            Complete lesson with all fragments
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{paper.metadata.arxiv_id}:{request.difficulty}"
        cached = self.cache.get("lessons", cache_key)
        if cached:
            logger.info(f"Lesson cache hit: {paper.metadata.arxiv_id}")
            return FullLesson(**cached)
        
        # Generate fragments
        fragments = []
        sections = paper.sections
        
        if request.max_sections:
            sections = sections[:request.max_sections]
        
        for i, section in enumerate(sections):
            next_section = sections[i + 1] if i + 1 < len(sections) else None
            
            fragment = self._generate_fragment(
                section=section,
                next_section_name=next_section.name if next_section else None,
                request=request,
                order=i
            )
            fragments.append(fragment)
            logger.debug(f"Generated fragment {i + 1}/{len(sections)}: {section.name}")
        
        # Create full lesson
        lesson = FullLesson(
            paper_id=paper.metadata.arxiv_id,
            paper_title=paper.metadata.title,
            paper_url=str(paper.metadata.url),
            query=request.query,
            fragments=fragments,
            difficulty=request.difficulty,
            generation_time_seconds=time.time() - start_time
        )
        
        # Cache result
        self.cache.set("lessons", cache_key, lesson.model_dump(mode='json'))
        
        logger.info(
            f"Generated lesson for {paper.metadata.arxiv_id}: "
            f"{len(fragments)} sections, {lesson.total_read_time} min read time"
        )
        
        return lesson
    
    async def generate_lesson_streaming(
        self,
        paper: ParsedPaper,
        request: LessonRequest
    ) -> AsyncGenerator[StreamingLessonChunk, None]:
        """
        Generate lesson with streaming responses.
        
        Yields chunks as sections are generated.
        """
        start_time = time.time()
        
        # Send metadata first
        yield StreamingLessonChunk(
            type="metadata",
            data={
                "paper_id": paper.metadata.arxiv_id,
                "paper_title": paper.metadata.title,
                "paper_url": str(paper.metadata.url),
                "total_sections": len(paper.sections),
            }
        )
        
        sections = paper.sections
        if request.max_sections:
            sections = sections[:request.max_sections]
        
        for i, section in enumerate(sections):
            try:
                next_section = sections[i + 1] if i + 1 < len(sections) else None
                
                content = await self._generate_fragment_async(
                    section=section,
                    next_section_name=next_section.name if next_section else None,
                    request=request
                )
                
                yield StreamingLessonChunk(
                    type="section",
                    data={
                        "name": section.name,
                        "content": content,
                        "order": i,
                        "progress": (i + 1) / len(sections)
                    }
                )
            except Exception as e:
                logger.error(f"Error generating section {section.name}: {e}")
                yield StreamingLessonChunk(
                    type="error",
                    data={"message": f"Failed to generate section: {section.name}"}
                )
        
        # Send completion
        yield StreamingLessonChunk(
            type="done",
            data={
                "total_time_seconds": time.time() - start_time,
                "sections_generated": len(sections)
            }
        )
    
    def _generate_fragment(
        self,
        section: PaperSection,
        next_section_name: Optional[str],
        request: LessonRequest,
        order: int
    ) -> LessonFragment:
        """Generate a single lesson fragment."""
        prompt = self._build_prompt(section, next_section_name, request)
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.lesson_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            raise LessonGenerationError(f"Failed to generate lesson: {e}", section.name)
        
        return LessonFragment(
            section_name=section.name,
            content=content,
            order=order,
            has_math=self._contains_math(content),
            has_code=self._contains_code(content)
        )
    
    async def _generate_fragment_async(
        self,
        section: PaperSection,
        next_section_name: Optional[str],
        request: LessonRequest
    ) -> str:
        """Generate a single lesson fragment asynchronously."""
        prompt = self._build_prompt(section, next_section_name, request)
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.settings.lesson_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LessonGenerationError(f"Failed to generate lesson: {e}", section.name)
    
    def _build_prompt(
        self,
        section: PaperSection,
        next_section_name: Optional[str],
        request: LessonRequest
    ) -> str:
        """Build the prompt for lesson generation."""
        difficulty_instructions = {
            LessonDifficulty.BEGINNER: "Use simple language, avoid jargon, and explain everything from first principles.",
            LessonDifficulty.INTERMEDIATE: "Assume basic ML/CS knowledge but explain advanced concepts clearly.",
            LessonDifficulty.ADVANCED: "Be concise and technical, focusing on nuances and advanced insights."
        }
        
        prompt = f"""You are an expert teacher converting a research paper section into a {request.difficulty.value}-friendly lesson.

Section: "{section.name}"

Content:
{section.content}

Instructions:
- {difficulty_instructions[request.difficulty]}
"""
        
        if request.include_math:
            prompt += "- Break down any mathematical concepts step by step.\n"
        else:
            prompt += "- Minimize mathematical notation, focus on intuition.\n"
        
        if request.include_examples:
            prompt += "- Include concrete examples and analogies to illustrate concepts.\n"
        
        if next_section_name:
            prompt += f'\n- End with a smooth transition to the next section: "{next_section_name}".\n'
        
        prompt += "\nGenerate the lesson fragment now:"
        
        return prompt
    
    def _contains_math(self, content: str) -> bool:
        """Check if content contains mathematical notation."""
        math_indicators = ['$', '\\frac', '\\sum', '\\int', '∑', '∫', '√', 'equation']
        return any(ind in content.lower() for ind in math_indicators)
    
    def _contains_code(self, content: str) -> bool:
        """Check if content contains code."""
        code_indicators = ['```', 'def ', 'import ', 'class ', 'function']
        return any(ind in content for ind in code_indicators)
    
    def generate_single_section_lesson(
        self,
        section_name: str,
        section_text: str,
        next_section_name: Optional[str] = None,
        difficulty: LessonDifficulty = LessonDifficulty.BEGINNER
    ) -> str:
        """
        Generate a lesson for a single section (backwards compatible).
        
        Args:
            section_name: Name of the section
            section_text: Section content
            next_section_name: Next section for transition
            difficulty: Lesson difficulty
            
        Returns:
            Generated lesson text
        """
        section = PaperSection(name=section_name, content=section_text, order=0)
        request = LessonRequest(
            query="",
            difficulty=difficulty,
            include_examples=True,
            include_math=True
        )
        
        fragment = self._generate_fragment(section, next_section_name, request, 0)
        return fragment.content


# Singleton instance
_lesson_service: Optional[LessonService] = None


def get_lesson_service() -> LessonService:
    """Get the global lesson service instance."""
    global _lesson_service
    if _lesson_service is None:
        _lesson_service = LessonService()
    return _lesson_service
