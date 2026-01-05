// src/components/MessageList.tsx
import { useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';
import { Loader2 } from 'lucide-react';
import type { Message } from '@/hooks/useChat';

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
}

export function MessageList({ messages, isLoading }: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} />
      ))}
      {isLoading && messages.length > 0 && messages[messages.length-1].role === 'user' && (
        <div className="flex justify-start">
            <div className="flex items-center gap-2 p-3 rounded-lg bg-muted">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Thinking...</span>
            </div>
        </div>
      )}
      <div ref={scrollRef} />
    </div>
  );
}