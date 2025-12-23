import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { 
  Code, 
  ExternalLink, 
  X,
  Tag,
  BarChart3,
  RefreshCw
} from 'lucide-react'
import { Card, CardContent, CardHeader } from './Card'
import Button from './Button'
import { Problem } from '@/lib/api'
import { cn } from '@/lib/utils'

interface ProblemDisplayProps {
  problem: Problem
  onClose: () => void
  onNewProblem: () => void
  isLoading: boolean
}

export default function ProblemDisplay({ 
  problem, 
  onClose, 
  onNewProblem,
  isLoading 
}: ProblemDisplayProps) {
  const difficultyColors = {
    Easy: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
    Medium: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300',
    Hard: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300',
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 overflow-y-auto bg-black/50 backdrop-blur-sm"
    >
      <div className="min-h-screen px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 50, scale: 0.95 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="max-w-4xl mx-auto"
        >
          <Card className="shadow-2xl">
            {/* Header */}
            <CardHeader className="relative">
              <div className="flex items-start justify-between">
                <div className="flex-1 pr-8">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-2">
                    <Code className="w-4 h-4" />
                    <span>Coding Challenge</span>
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
                    {problem.title}
                  </h2>
                  <div className="flex flex-wrap items-center gap-3">
                    <span className={cn(
                      'px-3 py-1 rounded-full text-sm font-medium',
                      difficultyColors[problem.difficulty]
                    )}>
                      {problem.difficulty}
                    </span>
                    {problem.acceptance_rate && (
                      <div className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400">
                        <BarChart3 className="w-4 h-4" />
                        <span>{problem.acceptance_rate.toFixed(1)}% acceptance</span>
                      </div>
                    )}
                    <a
                      href={problem.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-sm text-primary-500 hover:text-primary-600 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Solve on LeetCode</span>
                    </a>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onNewProblem}
                    isLoading={isLoading}
                  >
                    <RefreshCw className="w-4 h-4 mr-1" />
                    New Problem
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onClose}
                    className="text-gray-500"
                  >
                    <X className="w-5 h-5" />
                  </Button>
                </div>
              </div>
            </CardHeader>

            {/* Topics */}
            {problem.topics.length > 0 && (
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
                <div className="flex items-center gap-2 mb-2">
                  <Tag className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Related Topics
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {problem.topics.map((topic) => (
                    <span
                      key={topic}
                      className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Problem Statement */}
            <CardContent>
              <div className="prose-custom">
                <h3 className="text-lg font-semibold mb-4">Problem Statement</h3>
                <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl p-4 font-mono text-sm whitespace-pre-wrap">
                  {problem.statement}
                </div>
              </div>

              {/* Hints */}
              {problem.hints && problem.hints.length > 0 && (
                <div className="mt-6">
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
                    Hints (click to reveal)
                  </h4>
                  <div className="space-y-2">
                    {problem.hints.map((hint, index) => (
                      <HintItem key={index} hint={hint} index={index} />
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  )
}

function HintItem({ hint, index }: { hint: string; index: number }) {
  const [revealed, setRevealed] = useState(false)

  return (
    <button
      onClick={() => setRevealed(!revealed)}
      className="w-full text-left p-3 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
          Hint {index + 1}
        </span>
        <span className="text-xs text-gray-400">
          {revealed ? 'Click to hide' : 'Click to reveal'}
        </span>
      </div>
      {revealed && (
        <motion.p
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="mt-2 text-sm text-gray-700 dark:text-gray-300"
        >
          {hint}
        </motion.p>
      )}
    </button>
  )
}

// Need to import useState for HintItem
import { useState } from 'react'
