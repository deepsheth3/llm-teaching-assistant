import { motion } from 'framer-motion'
import { Sparkles, ArrowRight, RotateCcw } from 'lucide-react'
import Button from './Button'

interface PromptOptionsProps {
  original: string
  prompts: string[]
  onSelect: (prompt: string) => void
  onBack: () => void
  isLoading: boolean
}

export default function PromptOptions({
  original,
  prompts,
  onSelect,
  onBack,
  isLoading
}: PromptOptionsProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-2xl mx-auto"
    >
      {/* Original query */}
      <div className="text-center mb-8">
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
          You asked:
        </p>
        <p className="text-xl font-medium text-gray-900 dark:text-white">
          "{original}"
        </p>
      </div>

      {/* Prompt options */}
      <div className="mb-8">
        <p className="text-sm text-gray-500 dark:text-gray-400 text-center mb-4 flex items-center justify-center gap-2">
          <Sparkles className="w-4 h-4" />
          Choose a clearer version:
        </p>

        <div className="space-y-3">
          {prompts.map((prompt, index) => (
            <motion.button
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => onSelect(prompt)}
              disabled={isLoading}
              className="w-full p-4 text-left rounded-xl border-2 border-gray-200 dark:border-gray-700
                         bg-white dark:bg-gray-800
                         hover:border-primary-500 hover:bg-primary-50 dark:hover:bg-primary-900/50
                         focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
                         transition-all duration-200
                         disabled:opacity-50 disabled:cursor-not-allowed
                         group"
            >
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <span className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900 
                                   text-primary-600 dark:text-primary-400 
                                   flex items-center justify-center text-sm font-medium">
                    {index + 1}
                  </span>
                  <span className="text-gray-900 dark:text-white">
                    {prompt}
                  </span>
                </div>
                <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-primary-500 
                                       group-hover:translate-x-1 transition-all flex-shrink-0" />
              </div>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Back button */}
      <div className="text-center">
        <Button 
          variant="ghost" 
          onClick={onBack} 
          disabled={isLoading}
          className="text-gray-500"
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          Start over
        </Button>
      </div>
    </motion.div>
  )
}
