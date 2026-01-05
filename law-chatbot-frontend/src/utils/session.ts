import { v4 as uuidv4 } from 'uuid';

const SESSION_ID_KEY = 'law_chatbot_session_id';

/**
 * Get or create session ID for this browser
 */
export const getSessionId = (): string => {
  let sessionId = localStorage.getItem(SESSION_ID_KEY);
  
  if (!sessionId) {
    sessionId = uuidv4();
    localStorage.setItem(SESSION_ID_KEY, sessionId);
  }
  
  return sessionId;
};

/**
 * Clear session ID (for testing/logout)
 */
export const clearSessionId = (): void => {
  localStorage.removeItem(SESSION_ID_KEY);
};
