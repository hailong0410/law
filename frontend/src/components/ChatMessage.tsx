// src/components/ChatMessage.tsx
import { cn } from '@/lib/utils';
import { Bot, User, FileWarning } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import type { Message } from '@/hooks/useChat';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  return (
    <div
      className={cn('flex items-start gap-3', {
        'justify-end': isUser,
        'justify-center': isSystem,
      })}
    >
      {!isUser && !isSystem && (
        <Avatar className="h-8 w-8">
          <AvatarImage src="" alt="Bot" />
          <AvatarFallback>
            <Bot className="h-5 w-5" />
          </AvatarFallback>
        </Avatar>
      )}

      {isSystem ? (
        <div className="text-center text-xs text-muted-foreground p-2 rounded-lg bg-secondary">
          <FileWarning className="inline-block h-4 w-4 mr-2" />
          {message.content}
        </div>
      ) : (
        <div
          className={cn('max-w-[75%] rounded-lg p-3 text-sm break-words', {
            'bg-primary text-primary-foreground': isUser,
            'bg-muted': !isUser,
          })}
        >
          {message.content}
        </div>
      )}

      {isUser && (
        <Avatar className="h-8 w-8">
          <AvatarImage src="" alt="User" />
          <AvatarFallback>
            <User className="h-5 w-5" />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  );
}