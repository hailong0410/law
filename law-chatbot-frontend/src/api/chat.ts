import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ChatMessage {
    session_id: string;
    chat_content: string;
    time: string;
    role: 'user' | 'assistant';
    _id?: string;
}

export interface ChatRequest {
    session_id: string;
    chat_content: string;
}

export interface ChatResponse {
    status: string;
    session_id: string;
}

/**
 * Send a chat message to the backend
 */
export const sendChatMessage = async (sessionId: string, content: string): Promise<ChatResponse> => {
    const response = await axios.post<ChatResponse>(`${API_BASE_URL}/chat`, {
        session_id: sessionId,
        chat_content: content,
    });
    return response.data;
};

/**
 * Get all chat history for a session
 */
export const getChatHistory = async (sessionId: string): Promise<ChatMessage[]> => {
    const response = await axios.get<ChatMessage[]>(`${API_BASE_URL}/take_all_chat`, {
        params: { session_id: sessionId },
    });
    return response.data;
};

/**
 * Create SSE connection for real-time messages
 */
export const createSSEConnection = (sessionId: string): EventSource => {
    return new EventSource(`${API_BASE_URL}/stream?session_id=${sessionId}`);
};
