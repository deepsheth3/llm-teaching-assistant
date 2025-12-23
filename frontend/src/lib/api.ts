const API_BASE = '/api/v1'

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

// Non-streaming lesson generation
export async function generateLesson(request: LessonRequest): Promise<LessonResponse> {
  const response = await fetch(`${API_BASE}/teach`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return response.json()
}

// Streaming lesson generation
export async function* generateLessonStream(
  request: LessonRequest
): AsyncGenerator<StreamChunk> {
  const response = await fetch(`${API_BASE}/teach/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error('Failed to start streaming')
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No reader available')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6))
          yield data as StreamChunk
        } catch {
          // Skip invalid JSON
        }
      }
    }
  }
}

// Get random LeetCode problem
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

// Health check
export async function checkHealth(): Promise<{ status: string; version: string }> {
  const response = await fetch('/health')
  return response.json()
}

// Search papers
export async function searchPapers(query: string, topK: number = 5) {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  })
  return response.json()
}
