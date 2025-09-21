import { useState } from 'react';
import '../App.css';

function ConversationPanel({ messages, onSendQuestion, isSending }) {
  const [question, setQuestion] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      return;
    }
    await onSendQuestion(trimmed);
    setQuestion('');
  };

  return (
    <section className="section conversation-section">
      <div className="conversation-card">
        <h2>Ask about your documents</h2>
        <p className="description">
          We’ll retrieve the most relevant passages and respond with cited answers using your indexed files.
        </p>

        <div className="message-list">
          {messages.length === 0 ? (
            <div className="empty-state">
              Upload documents and ask a question to start a conversation.
            </div>
          ) : (
            messages.map((message) => (
              <div key={message.id} className={`message ${message.role}`}>
                {message.content}
                {message.role === 'assistant' && message.sources?.length ? (
                  <div className="citations">
                    {message.sources.map((source, index) => (
                      <div key={`${source.source}-${index}`} className="citation">
                        <strong>{source.source}</strong>
                        <span>{source.preview}</span>
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
            ))
          )}
        </div>

        <form className="chat-form" onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Ask a question about your documents"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            disabled={isSending}
          />
          <button type="submit" disabled={isSending || !question.trim()}>
            {isSending ? 'Thinking…' : 'Send'}
          </button>
        </form>
      </div>
    </section>
  );
}

export default ConversationPanel;
