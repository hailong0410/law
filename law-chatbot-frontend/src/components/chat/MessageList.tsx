import { useEffect, useRef } from 'react';
import type { ChatMessage as ChatMessageType } from '@/api/chat';
import { Message } from './Message';
import { ScrollArea } from '@/components/ui/scroll-area';

interface MessageListProps {
    messages: ChatMessageType[];
    loading?: boolean;
}

export const MessageList = ({ messages, loading }: MessageListProps) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground">Loading chat history...</p>
            </div>
        );
    }

    if (messages.length === 0) {
        return (
            <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground">No messages yet. Start chatting!</p>
            </div>
        );
    }

    return (
        <ScrollArea className="flex-1 p-4">
            <div ref={scrollRef} className="space-y-2">
                {messages.map((message, index) => (
                    <Message key={message._id || index} message={message} />
                ))}
            </div>
        </ScrollArea>
    );
};
