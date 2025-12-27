const API_BASE = (import.meta as any).env?.VITE_API_URL 
  ? `${(import.meta as any).env.VITE_API_URL}/api/v1`
  : '/api/v1'

export interface LessonRequest {
  query: string
  difficulty?: 'beginner' | 'intermediate' | 'advanced'
  include_examples?: boolean
  include_math?: boolean
  max_sections?: number
}

export interface LessonFragment {
  section_name: string
  content: string
  order: number
  estimated_read_time: number
}

export interface Lesson {
  paper_id: string
  paper_title: string
  paper_url: string
  query: string
  fragments: LessonFragment[]
  total_read_time: number
  generation_time_seconds: number
}

export interface LessonResponse {
  success: boolean
  lesson?: Lesson
  error?: string
  processing_time_ms: number
}

export interface Problem {
  title: string
  slug: string
  difficulty: 'Easy' | 'Medium' | 'Hard'
  statement: string
  url: string
  topics: string[]
}

export interface ProblemResponse {
  success: boolean
  problem?: Problem
  error?: string
  processing_time_ms: number
}

export interface StreamChunk {
  type: 'metadata' | 'section' | 'done' | 'error'
  data: Record<string, unknown>
}

// ============ NEW: Clarify Types ============

export interface ClarifyResponse {
  original: string
  prompts: string[]
}

// ============ API Functions ============

export async function clarifyQuery(query: string): Promise<ClarifyResponse> {
  const response = await fetch(`${API_BASE}/clarify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
  
  if (!response.ok) {
    throw new Error('Failed to clarify query')
  }
  
  return response.json()
}

export async function generateLesson(request: LessonRequest): Promise<LessonResponse> {
  const response = await fetch(`${API_BASE}/teach`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return response.json()
}

export async function getRandomProblem(
  difficulties: string[] = ['Medium', 'Hard']
): Promise<ProblemResponse> {
  const response = await fetch(`${API_BASE}/leetcode/random`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ difficulties, exclude_premium: true }),
  })
  return response.json()
}

export async function searchPapers(query: string, topK: number = 5) {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  })
  return response.json()
}
