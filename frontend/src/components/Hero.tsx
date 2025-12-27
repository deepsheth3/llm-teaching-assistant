import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, ArrowRight, BookOpen, Code } from 'lucide-react'
import Button from './Button'
import { Textarea } from './Input'
import PromptOptions from './PromptOptions'
import { clarifyQuery } from '@/lib/api'

interface HeroProps {
  onSubmit: (query: string, mode: 'learn' | 'code') => void
  isLoading: boolean
}

type HeroState =
  | { step: 'input' }
  | { step: 'clarifying' }
  | { step: 'options'; original: string; prompts: string[] }

export default function Hero({ onSubmit, isLoading }: HeroProps) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'learn' | 'code'>('learn')
  const [state, setState] = useState<HeroState>({ step: 'input' })

  const handleSubmitQuery = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    if (mode === 'code') {
      // Skip clarification for code mode
      onSubmit(query, 'code')
      return
    }

    // Learn mode: clarify first
    setState({ step: 'clarifying' })

    try {
      const response = await clarifyQuery(query)
      setState({
        step: 'options',
        original: response.original,
        prompts: response.prompts
      })
    } catch (error) {
      console.error('Clarification failed:', error)
      // Fallback: use original query
      onSubmit(query, 'learn')
    }
  }

  const handleSelectPrompt = (selectedPrompt: string) => {
    onSubmit(selectedPrompt, 'learn')
  }

  const handleBack = () => {
    setState({ step: 'input' })
  }

  const suggestions = [
    'Explain attention mechanisms',
    'What is BERT?',
    'GPT vs BERT difference',
    'How does LoRA work?',
  ]

  return (
    <section className="relative min-h-screen flex items-center justify-center px-4 pt-24 pb-12">
      {/* Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-4xl mx-auto text-center w-full">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-50 dark:bg-primary-950 text-primary-600 dark:text-primary-400 text-sm font-medium mb-6">
            <Sparkles className="w-4 h-4" />
            <span>AI-Powered Learning</span>
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight mb-6">
            Learn AI Research
            <br />
            <span className="text-gradient">The Easy Way</span>
          </h1>

          <p className="text-lg sm:text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Transform complex research papers into beginner-friendly lessons.
          </p>
        </motion.div>

        {/* Dynamic Content */}
        <AnimatePresence mode="wait">
          {/* Step 1: Input */}
          {state.step === 'input' && (
            <motion.div
              key="input"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              {/* Mode Toggle */}
              <div className="flex justify-center gap-2 mb-6">
                <button
                  onClick={() => setMode('learn')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
                    mode === 'learn'
                      ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                  }`}
                >
                  <BookOpen className="w-4 h-4" />
                  Learn
                </button>
                <button
                  onClick={() => setMode('code')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
                    mode === 'code'
                      ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                  }`}
                >
                  <Code className="w-4 h-4" />
                  Practice
                </button>
              </div>

              {/* Input Form */}
              <form onSubmit={handleSubmitQuery} className="max-w-2xl mx-auto">
                <div className="relative">
                  <Textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={
                      mode === 'learn'
                        ? 'What do you want to learn? (e.g., "What is GPT?")'
                        : 'Describe what you want to practice...'
                    }
                    rows={3}
                    className="pr-24 text-lg"
                  />
                  <div className="absolute right-2 bottom-2">
                    <Button
                      type="submit"
                      disabled={!query.trim() || isLoading}
                      className="rounded-xl"
                    >
                      {mode === 'learn' ? 'Next' : 'Go'}
                      <ArrowRight className="w-4 h-4 ml-1" />
                    </Button>
                  </div>
                </div>
              </form>

              {/* Suggestions */}
              {mode === 'learn' && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                  className="mt-6 flex flex-wrap justify-center gap-2"
                >
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    Try:
                  </span>
                  {suggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => setQuery(suggestion)}
                      className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 
                                 text-gray-600 dark:text-gray-300 
                                 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </motion.div>
              )}
            </motion.div>
          )}

          {/* Step 2: Clarifying */}
          {state.step === 'clarifying' && (
            <motion.div
              key="clarifying"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4 py-12"
            >
              <div className="relative">
                <div className="w-12 h-12 rounded-full border-4 border-primary-200 dark:border-primary-800" />
                <div className="absolute inset-0 w-12 h-12 rounded-full border-4 border-transparent border-t-primary-500 animate-spin" />
              </div>
              <p className="text-gray-600 dark:text-gray-400">
                Generating clearer prompts...
              </p>
            </motion.div>
          )}

          {/* Step 3: Options */}
          {state.step === 'options' && (
            <PromptOptions
              key="options"
              original={state.original}
              prompts={state.prompts}
              onSelect={handleSelectPrompt}
              onBack={handleBack}
              isLoading={isLoading}
            />
          )}
        </AnimatePresence>
      </div>
    </section>
  )
}
