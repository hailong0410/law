import { useState } from 'react';
import { useChat } from '@/hooks/useChat';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from './ui/card';

interface ChatLayoutProps {
  sessionId: string;
}

export function ChatLayout({ sessionId }: ChatLayoutProps) {
  const {
    messages,
    input,
    isLoading,
    handleInputChange,
    sendMessage,
    uploadFile,
  } = useChat(sessionId);
  
  const [isTextRagEnabled, setIsTextRagEnabled] = useState(false);

  return (
    <Card className="h-full w-full max-w-3xl flex flex-col shadow-xl">
      <CardHeader className="border-b flex flex-row items-center justify-between p-4">
        <CardTitle>Multimodal AI Chatbot</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 p-0 overflow-hidden">
        <MessageList messages={messages} isLoading={isLoading} />
      </CardContent>
      <CardFooter className="p-0 border-t">
        <ChatInput
          input={input}
          isLoading={isLoading}
          handleInputChange={handleInputChange}
          sendMessage={sendMessage}
          uploadFile={uploadFile}
          isRagEnabled={isTextRagEnabled}
          onRagToggle={setIsTextRagEnabled}
        />
      </CardFooter>
    </Card>
  );
}