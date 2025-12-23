import { useState } from 'react';
import type { FormEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send } from 'lucide-react';

interface ChatInputProps {
    onSend: (message: string) => void;
    disabled?: boolean;
}

export const ChatInput = ({ onSend, disabled }: ChatInputProps) => {
    const [input, setInput] = useState('');

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();

        const trimmedInput = input.trim();
        if (!trimmedInput || disabled) return;

        onSend(trimmedInput);
        setInput('');
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="border-t p-4 bg-background">
            <div className="flex gap-2">
                <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
                    disabled={disabled}
                    className="min-h-[60px] max-h-[200px] resize-none"
                />
                <Button
                    type="submit"
                    disabled={disabled || !input.trim()}
                    size="icon"
                    className="h-[60px] w-[60px]"
                >
                    <Send className="h-5 w-5" />
                </Button>
            </div>
        </form>
    );
};
