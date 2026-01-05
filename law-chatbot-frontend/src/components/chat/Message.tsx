import type { ChatMessage } from '@/api/chat';
import { cn } from '@/lib/utils';

interface MessageProps {
    message: ChatMessage;
}

export const Message = ({ message }: MessageProps) => {
    const isUser = message.role === 'user';

    return (
        <div
            className={cn(
                'flex w-full mb-4',
                isUser ? 'justify-end' : 'justify-start'
            )}
        >
            <div
                className={cn(
                    'max-w-[70%] rounded-lg px-4 py-2',
                    isUser
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-foreground'
                )}
            >
                <p className="text-sm whitespace-pre-wrap break-words">
                    {message.chat_content}
                </p>
                <span className="text-xs opacity-70 mt-1 block">
                    {new Date(message.time).toLocaleTimeString()}
                </span>
            </div>
        </div>
    );
};
