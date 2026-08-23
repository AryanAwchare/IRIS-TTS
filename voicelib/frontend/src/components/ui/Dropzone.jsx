import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { CloudArrowUpIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline'
import { clsx } from 'clsx'

const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.webm', '.mp4']

const ACCEPT_ATTRIBUTE_STRING = '.mp3,.wav,.m4a,.aac,.ogg,.flac,.webm,.mp4,audio/*,audio/mpeg,audio/mp3,audio/x-mp3,audio/mpeg3,audio/x-mpeg-3'

const DEFAULT_ACCEPT = {
  'audio/mpeg': ['.mp3', '.mpga', '.mpega', '.mp2'],
  'audio/mp3': ['.mp3'],
  'audio/x-mp3': ['.mp3'],
  'audio/mpeg3': ['.mp3'],
  'audio/x-mpeg-3': ['.mp3'],
  'audio/x-mpeg': ['.mp3'],
  'audio/wav': ['.wav'],
  'audio/x-wav': ['.wav'],
  'audio/wave': ['.wav'],
  'audio/m4a': ['.m4a'],
  'audio/x-m4a': ['.m4a'],
  'audio/aac': ['.aac'],
  'audio/x-aac': ['.aac'],
  'audio/ogg': ['.ogg'],
  'audio/flac': ['.flac'],
  'audio/webm': ['.webm'],
  'audio/mp4': ['.mp4'],
  'audio/*': ALLOWED_EXTENSIONS,
}

export function Dropzone({
  onFileAccepted,
  error,
  accept = DEFAULT_ACCEPT,
}) {
  const [file, setFile] = useState(null)
  const [clientError, setClientError] = useState(null)

  const onDrop = useCallback(async (acceptedFiles, fileRejections) => {
    let f = acceptedFiles[0]

    // Handle react-dropzone rejections (e.g. unknown/missing OS MIME types for .mp3)
    if (!f && fileRejections && fileRejections.length > 0) {
      const rejectedFile = fileRejections[0]?.file
      if (rejectedFile) {
        const ext = '.' + (rejectedFile.name.split('.').pop() || '').toLowerCase()
        if (ALLOWED_EXTENSIONS.includes(ext)) {
          f = rejectedFile
        } else {
          setClientError(`Unsupported file format (${ext}). Please upload an MP3, WAV, M4A, AAC, OGG, or FLAC file.`)
          setFile(null)
          onFileAccepted?.(null)
          return
        }
      }
    }

    if (!f) return
    setClientError(null)

    // Client-side size check (20 MB)
    if (f.size > 20 * 1024 * 1024) {
      setClientError('File exceeds 20 MB. Please trim the audio.')
      setFile(null)
      onFileAccepted?.(null)
      return
    }

    // Client-side duration check via <audio> element
    try {
      const url = URL.createObjectURL(f)
      const audio = new Audio(url)
      await new Promise((resolve) => {
        audio.onloadedmetadata = () => resolve(true)
        audio.onerror = () => resolve(false)
        setTimeout(() => resolve(false), 2000)
      })
      URL.revokeObjectURL(url)

      // Only reject if duration is finite and strictly > 31 seconds
      if (Number.isFinite(audio.duration) && audio.duration > 31) {
        setClientError(`Audio is ${Math.round(audio.duration)}s long. Maximum allowed duration is 30 seconds.`)
        setFile(null)
        onFileAccepted?.(null)
        return
      }
    } catch {
      // If client audio metadata inspection fails, let backend handle validation
    }

    setFile(f)
    onFileAccepted?.(f)
  }, [onFileAccepted])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    multiple: false,
    maxFiles: 1,
  })

  const displayError = clientError || error

  const formatFileSize = (bytes) => {
    if (bytes >= 1024 * 1024) {
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
    }
    return (bytes / 1024).toFixed(0) + ' KB'
  }

  return (
    <div className="space-y-2">
      <div
        {...getRootProps()}
        className={clsx(
          'relative flex flex-col items-center justify-center gap-3',
          'rounded-3xl border-2 border-dashed p-10 cursor-pointer',
          'transition-all duration-500 ease-spring',
          isDragActive
            ? 'border-primary-500 bg-primary-500/10 scale-[1.01]'
            : file
            ? 'border-emerald-500/40 bg-emerald-500/5'
            : 'border-white/[0.1] bg-white/[0.02] hover:border-white/[0.2] hover:bg-white/[0.04]'
        )}
      >
        <input {...getInputProps({ accept: ACCEPT_ATTRIBUTE_STRING })} />

        {file ? (
          <>
            <CheckCircleIcon className="w-12 h-12 text-emerald-400 shrink-0" />
            <div className="text-center">
              <p className="text-base font-bold text-emerald-300">{file.name}</p>
              <p className="text-sm font-medium text-surface-200 mt-1">
                {formatFileSize(file.size)} — click or drop to replace
              </p>
            </div>
          </>
        ) : (
          <>
            <div className="w-16 h-16 rounded-full bg-primary-500/15 border border-primary-500/30 flex items-center justify-center shrink-0 shadow-glow-sm">
              <CloudArrowUpIcon className="w-8 h-8 text-primary-400" />
            </div>
            <div className="text-center">
              <p className="text-base sm:text-lg font-bold text-white">
                {isDragActive ? 'Drop your audio file here' : 'Drop audio file here'}
              </p>
              <p className="text-sm font-medium text-surface-200 mt-1.5">
                MP3, WAV, M4A, OGG, FLAC · Max 30s · Max 20 MB
              </p>
              <p className="text-sm font-semibold text-primary-300 mt-2.5">Or click to browse files</p>
            </div>
          </>
        )}
      </div>

      {displayError && (
        <div className="flex items-center gap-2 text-xs text-red-400">
          <XCircleIcon className="w-4 h-4 shrink-0" />
          {displayError}
        </div>
      )}
    </div>
  )
}

