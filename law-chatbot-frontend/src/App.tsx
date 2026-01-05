import { Toaster } from 'react-hot-toast';
import { ChatContainer } from './components/chat/ChatContainer';

function App() {
  return (
    <>
      <ChatContainer />
      <Toaster position="top-right" />
    </>
  );
}

export default App;