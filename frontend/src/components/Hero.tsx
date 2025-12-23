import { useState } from 'react'
import { motion } from 'framer-motion'
import { Sparkles, ArrowRight, BookOpen, Code, Search } from 'lucide-react'
import Button from './Button'
import { Textarea } from './Input'

interface HeroProps {
  onSubmit: (query: string, mode: 'learn' | 'code') => void
  isLoading: boolean
}

export default function Hero({ onSubmit, isLoading }: HeroProps) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'learn' | 'code'>('learn')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSubmit(query, mode)
    }
  }

  const suggestions = [
    'Explain attention mechanisms in transformers',
    'How does BERT pre-training work?',
    'What is LoRA fine-tuning?',
    'Explain the GPT architecture',
  ]

  return (
    <section className="relative min-h-screen flex items-center justify-center px-4 pt-24 pb-12">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-4xl mx-auto text-center">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-50 dark:bg-primary-950 text-primary-600 dark:text-primary-400 text-sm font-medium mb-6"
        >
          <Sparkles className="w-4 h-4" />
          <span>AI-Powered Learning</span>
        </motion.div>

        {/* Heading */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight mb-6"
        >
          Learn AI Research
          <br />
          <span className="text-gradient">The Easy Way</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-lg sm:text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto"
        >
          Transform complex research papers into beginner-friendly lessons.
          Practice coding with curated LeetCode problems.
        </motion.p>

        {/* Mode switcher */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="flex justify-center gap-2 mb-6"
        >
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
        </motion.div>

        {/* Input form */}
        <motion.form
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          onSubmit={handleSubmit}
          className="relative max-w-2xl mx-auto"
        >
          <div className="relative">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                mode === 'learn'
                  ? 'What would you like to learn about? (e.g., "Explain transformers")'
                  : 'Describe what you want to practice...'
              }
              rows={3}
              className="pr-24 text-lg"
            />
            <div className="absolute right-2 bottom-2">
              <Button
                type="submit"
                isLoading={isLoading}
                disabled={!query.trim()}
                className="rounded-xl"
              >
                {isLoading ? (
                  'Generating...'
                ) : (
                  <>
                    Go
                    <ArrowRight className="w-4 h-4 ml-1" />
                  </>
                )}
              </Button>
            </div>
          </div>
        </motion.form>

        {/* Suggestions */}
        {mode === 'learn' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="mt-6 flex flex-wrap justify-center gap-2"
          >
            <span className="text-sm text-gray-500 dark:text-gray-400 mr-2">
              Try:
            </span>
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setQuery(suggestion)}
                className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </motion.div>
        )}
      </div>
    </section>
  )
}
