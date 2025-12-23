import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { 
  BookOpen, 
  Clock, 
  ExternalLink, 
  ChevronDown, 
  ChevronUp,
  Copy,
  Check,
  X
} from 'lucide-react'
import { Card, CardContent, CardHeader } from './Card'
import Button from './Button'
import { Lesson, LessonFragment } from '@/lib/api'
import { cn, formatReadTime } from '@/lib/utils'

interface LessonDisplayProps {
  lesson: Lesson
  onClose: () => void
}

export default function LessonDisplay({ lesson, onClose }: LessonDisplayProps) {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(
    new Set(lesson.fragments.map((_, i) => i)) // All expanded by default
  )
  const [copied, setCopied] = useState(false)

  const toggleSection = (index: number) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedSections(newExpanded)
  }

  const copyLesson = async () => {
    const fullContent = lesson.fragments
      .map((f) => `## ${f.section_name}\n\n${f.content}`)
      .join('\n\n---\n\n')
    
    await navigator.clipboard.writeText(fullContent)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
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
                    <BookOpen className="w-4 h-4" />
                    <span>Lesson from research paper</span>
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    {lesson.paper_title}
                  </h2>
                  <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      <span>{formatReadTime(lesson.total_read_time)}</span>
                    </div>
                    <a
                      href={lesson.paper_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 hover:text-primary-500 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>View Paper</span>
                    </a>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyLesson}
                    className="text-gray-500"
                  >
                    {copied ? (
                      <Check className="w-4 h-4 text-green-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
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

            {/* Table of Contents */}
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
                Table of Contents
              </h3>
              <div className="flex flex-wrap gap-2">
                {lesson.fragments.map((fragment, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setExpandedSections(new Set([index]))
                      document.getElementById(`section-${index}`)?.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start',
                      })
                    }}
                    className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-primary-100 dark:hover:bg-primary-900 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
                  >
                    {fragment.section_name}
                  </button>
                ))}
              </div>
            </div>

            {/* Content */}
            <CardContent className="space-y-4">
              {lesson.fragments.map((fragment, index) => (
                <LessonSection
                  key={index}
                  fragment={fragment}
                  index={index}
                  isExpanded={expandedSections.has(index)}
                  onToggle={() => toggleSection(index)}
                />
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  )
}

interface LessonSectionProps {
  fragment: LessonFragment
  index: number
  isExpanded: boolean
  onToggle: () => void
}

function LessonSection({ fragment, index, isExpanded, onToggle }: LessonSectionProps) {
  return (
    <motion.div
      id={`section-${index}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden"
    >
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 flex items-center justify-between bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="w-6 h-6 rounded-full bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400 text-sm font-medium flex items-center justify-center">
            {index + 1}
          </span>
          <h3 className="font-semibold text-gray-900 dark:text-white capitalize">
            {fragment.section_name}
          </h3>
        </div>
        <div className="flex items-center gap-2 text-gray-500">
          <span className="text-sm">{formatReadTime(fragment.estimated_read_time)}</span>
          {isExpanded ? (
            <ChevronUp className="w-5 h-5" />
          ) : (
            <ChevronDown className="w-5 h-5" />
          )}
        </div>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-4 prose-custom">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {fragment.content}
              </ReactMarkdown>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
