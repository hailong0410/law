import { useEffect, useState, useCallback } from 'react';
import { createSSEConnection } from '../api/chat';

interface SSEMessage {
    content: string;
}

export const useSSE = (sessionId: string) => {
    const [message, setMessage] = useState<string | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!sessionId) return;

        let eventSource: EventSource | null = null;

        try {
            eventSource = createSSEConnection(sessionId);

            eventSource.onopen = () => {
                console.log('SSE connection established');
                setIsConnected(true);
                setError(null);
            };

            eventSource.onmessage = (event) => {
                try {
                    const data: SSEMessage = JSON.parse(event.data);
                    setMessage(data.content);
                } catch (err) {
                    console.error('Failed to parse SSE message:', err);
                }
            };

            eventSource.onerror = (err) => {
                console.error('SSE error:', err);
                setIsConnected(false);
                setError('Connection lost. Reconnecting...');
                eventSource?.close();
            };
        } catch (err) {
            console.error('Failed to create SSE connection:', err);
            setError('Failed to connect');
        }

        return () => {
            if (eventSource) {
                eventSource.close();
                setIsConnected(false);
            }
        };
    }, [sessionId]);

    const clearMessage = useCallback(() => {
        setMessage(null);
    }, []);

    return { message, isConnected, error, clearMessage };
};
