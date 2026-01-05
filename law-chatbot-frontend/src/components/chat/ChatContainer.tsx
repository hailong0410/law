import { useEffect, useState } from 'react';
import { getSessionId } from '@/utils/session';
import { sendChatMessage } from '@/api/chat';
import type { ChatMessage } from '@/api/chat';
import { useSSE } from '@/hooks/useSSE';
import { useChatHistory } from '@/hooks/useChatHistory';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { toast } from 'react-hot-toast';

export const ChatContainer = () => {
    const [sessionId] = useState(() => getSessionId());
    const [sending, setSending] = useState(false);

    const { messages, loading, addMessage } = useChatHistory(sessionId);
    const { message: sseMessage, isConnected, error, clearMessage } = useSSE(sessionId);

    // Handle incoming SSE messages
    useEffect(() => {
        if (sseMessage) {
            const newMessage: ChatMessage = {
                session_id: sessionId,
                chat_content: sseMessage,
                time: new Date().toISOString(),
                role: 'assistant',
            };
            addMessage(newMessage);
            clearMessage();
        }
    }, [sseMessage, sessionId, addMessage, clearMessage]);

    // Show error toast
    useEffect(() => {
        if (error) {
            toast.error(error);
        }
    }, [error]);

    const handleSendMessage = async (content: string) => {
        try {
            setSending(true);

            // Add user message to UI immediately
            const userMessage: ChatMessage = {
                session_id: sessionId,
                chat_content: content,
                time: new Date().toISOString(),
                role: 'user',
            };
            addMessage(userMessage);

            // Send to backend
            await sendChatMessage(sessionId, content);

            toast.success('Message sent');
        } catch (err) {
            console.error('Failed to send message:', err);
            toast.error('Failed to send message');
        } finally {
            setSending(false);
        }
    };

    return (
        <div className="h-screen flex flex-col">
            {/* Header */}
            <div className="border-b p-4 bg-background">
                <div className="flex items-center justify-between">
                    <h1 className="text-2xl font-bold">Law Chatbot</h1>
                    <div className="flex items-center gap-2">
                        <Badge variant={isConnected ? 'default' : 'destructive'}>
                            {isConnected ? 'Connected' : 'Disconnected'}
                        </Badge>
                        <Badge variant="outline" className="font-mono text-xs">
                            {sessionId.slice(0, 8)}...
                        </Badge>
                    </div>
                </div>
            </div>

            {/* Messages */}
            <Card className="flex-1 flex flex-col m-4 overflow-hidden">
                <MessageList messages={messages} loading={loading} />
                <ChatInput onSend={handleSendMessage} disabled={sending || !isConnected} />
            </Card>
        </div>
    );
};
