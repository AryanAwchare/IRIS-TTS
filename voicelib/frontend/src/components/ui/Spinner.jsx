import { clsx } from 'clsx'
import { ArrowPathIcon } from '@heroicons/react/24/outline'

export function Spinner({ size = 'md', className = '' }) {
  const sizes = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-8 h-8' }
  return (
    <ArrowPathIcon
      className={clsx('animate-spin-slow text-primary-500', sizes[size], className)}
    />
  )
}
