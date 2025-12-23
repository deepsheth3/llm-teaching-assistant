import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import { ThemeProvider } from '@/hooks/useTheme'
import { Header, Hero, LessonDisplay, ProblemDisplay, LoadingOverlay } from '@/components'
import { generateLesson, getRandomProblem, Lesson, Problem, LessonRequest } from '@/lib/api'

type ViewState = 
  | { type: 'home' }
  | { type: 'loading'; message: string }
  | { type: 'lesson'; lesson: Lesson }
  | { type: 'problem'; problem: Problem }
  | { type: 'error'; message: string }

export default function App() {
  const [viewState, setViewState] = useState<ViewState>({ type: 'home' })
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (query: string, mode: 'learn' | 'code') => {
    setIsLoading(true)

    try {
      if (mode === 'learn') {
        setViewState({ type: 'loading', message: 'Searching for relevant papers...' })
        
        const request: LessonRequest = {
          query,
          difficulty: 'beginner',
          include_examples: true,
          include_math: true,
          max_sections: 5,  // ← ADD THIS LINE
        }

        const response = await generateLesson(request)

        if (response.success && response.lesson) {
          setViewState({ type: 'lesson', lesson: response.lesson })
        } else {
          setViewState({ 
            type: 'error', 
            message: response.error || 'Failed to generate lesson' 
          })
        }
      } else {
        setViewState({ type: 'loading', message: 'Finding a coding challenge...' })
        
        const response = await getRandomProblem()

        if (response.success && response.problem) {
          setViewState({ type: 'problem', problem: response.problem })
        } else {
          setViewState({ 
            type: 'error', 
            message: response.error || 'Failed to fetch problem' 
          })
        }
      }
    } catch (error) {
      console.error('Error:', error)
      setViewState({ 
        type: 'error', 
        message: 'Something went wrong. Please try again.' 
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleClose = () => {
    setViewState({ type: 'home' })
  }

  const handleNewProblem = async () => {
    setIsLoading(true)
    try {
      const response = await getRandomProblem()
      if (response.success && response.problem) {
        setViewState({ type: 'problem', problem: response.problem })
      }
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 transition-colors">
        <Header />
        
        <main>
          <Hero onSubmit={handleSubmit} isLoading={isLoading} />
        </main>

        {/* Modals */}
        <AnimatePresence>
          {viewState.type === 'loading' && (
            <LoadingOverlay message={viewState.message} />
          )}

          {viewState.type === 'lesson' && (
            <LessonDisplay 
              lesson={viewState.lesson} 
              onClose={handleClose} 
            />
          )}

          {viewState.type === 'problem' && (
            <ProblemDisplay
              problem={viewState.problem}
              onClose={handleClose}
              onNewProblem={handleNewProblem}
              isLoading={isLoading}
            />
          )}
        </AnimatePresence>

        {/* Error Toast */}
        <AnimatePresence>
          {viewState.type === 'error' && (
            <div className="fixed bottom-4 right-4 z-50">
              <div className="bg-red-500 text-white px-4 py-3 rounded-xl shadow-lg flex items-center gap-3">
                <span>{viewState.message}</span>
                <button
                  onClick={handleClose}
                  className="text-white/80 hover:text-white"
                >
                  ✕
                </button>
              </div>
            </div>
          )}
        </AnimatePresence>
      </div>
    </ThemeProvider>
  )
}
