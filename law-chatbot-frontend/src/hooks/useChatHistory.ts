import { useCallback, useEffect, useState } from 'react';
import { getChatHistory } from '../api/chat';
import type { ChatMessage } from '../api/chat';

export const useChatHistory = (sessionId: string) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadHistory = useCallback(async () => {
        if (!sessionId) return;

        try {
            setLoading(true);
            setError(null);
            const history = await getChatHistory(sessionId);
            setMessages(history);
        } catch (err) {
            console.error('Failed to load chat history:', err);
            setError('Failed to load chat history');
        } finally {
            setLoading(false);
        }
    }, [sessionId]);

    useEffect(() => {
        loadHistory();
    }, [loadHistory]);

    const addMessage = useCallback((message: ChatMessage) => {
        setMessages((prev) => [...prev, message]);
    }, []);

    return { messages, loading, error, addMessage, reload: loadHistory };
};
