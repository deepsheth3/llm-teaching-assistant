import { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { motion, HTMLMotionProps } from 'framer-motion'

interface CardProps {
  children: ReactNode
  className?: string
  hover?: boolean
  glass?: boolean
}

export function Card({ children, className, hover = false, glass = false }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-2xl border',
        glass 
          ? 'glass glass-border' 
          : 'bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-800',
        hover && 'transition-all duration-300 hover:shadow-xl hover:shadow-primary-500/5 hover:-translate-y-1',
        className
      )}
    >
      {children}
    </div>
  )
}

export function CardHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('px-6 py-4 border-b border-gray-200 dark:border-gray-800', className)}>
      {children}
    </div>
  )
}

export function CardContent({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('px-6 py-4', className)}>
      {children}
    </div>
  )
}

export function CardFooter({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('px-6 py-4 border-t border-gray-200 dark:border-gray-800', className)}>
      {children}
    </div>
  )
}

// Animated card
export const MotionCard = motion(Card)
