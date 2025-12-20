// src/App.tsx
import { useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Toaster } from 'react-hot-toast';
import { ChatLayout } from '@/components/ChatLayout';

function App() {
  // Generate a unique session ID for the user's visit
  const [sessionId] = useState(uuidv4());

  return (
    <main className="flex h-screen w-screen items-center justify-center bg-secondary p-4">
      <Toaster position="top-center" />
      <ChatLayout sessionId={sessionId} />
    </main>
  );
}

export default App;