import { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const HISTORY_KEY = 'verdant-chat-history'
const ENDPOINT_KEY = 'verdant-chat-endpoint'
const defaultEndpoint = import.meta.env.VITE_CHAT_ENDPOINT || '/chat'

const welcome = {
  id: 'welcome',
  role: 'assistant',
  text: 'Hello! How can I help today?',
  timestamp: new Date().toISOString(),
}

function readHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(HISTORY_KEY))
    return Array.isArray(saved) && saved.length ? saved : [welcome]
  } catch {
    return [welcome]
  }
}

function getResponseText(data) {
  if (typeof data === 'string') return data
  if (!data || typeof data !== 'object') return 'The server returned an empty response.'
  return data.response ?? data.message ?? data.text ?? data.answer ?? data.content
    ?? 'Response received. Set the response mapping in getResponseText() once the API contract is finalized.'
}

function App() {
  const [messages, setMessages] = useState(readHistory)
  const [draft, setDraft] = useState('')
  const [endpoint, setEndpoint] = useState(() => localStorage.getItem(ENDPOINT_KEY) || defaultEndpoint)
  const [showSettings, setShowSettings] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const endRef = useRef(null)

  useEffect(() => localStorage.setItem(HISTORY_KEY, JSON.stringify(messages)), [messages])
  useEffect(() => localStorage.setItem(ENDPOINT_KEY, endpoint), [endpoint])
  useEffect(() => endRef.current?.scrollIntoView({ behavior: 'smooth' }), [messages, isLoading])

  async function sendMessage(event) {
    event?.preventDefault()
    const text = draft.trim()
    if (!text || isLoading) return

    const userMessage = { id: crypto.randomUUID(), role: 'user', text, timestamp: new Date().toISOString() }
    setMessages(current => [...current, userMessage])
    setDraft('')
    setError('')
    setIsLoading(true)

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      if (!response.ok) throw new Error(`Request failed (${response.status})`)
      const contentType = response.headers.get('content-type') || ''
      const payload = contentType.includes('application/json') ? await response.json() : await response.text()
      setMessages(current => [...current, {
        id: crypto.randomUUID(), role: 'assistant', text: getResponseText(payload), timestamp: new Date().toISOString(),
      }])
    } catch (requestError) {
      setError(requestError.message || 'Unable to reach the chat service.')
    } finally {
      setIsLoading(false)
    }
  }

  function clearChat() {
    setMessages([welcome])
    setError('')
  }

  return (
    <main className="app-shell">
      <section className="chat-card" aria-label="Chat application">
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">✦</span>
            <div><p>Verdant</p><span>AI assistant</span></div>
          </div>
          <div className="header-actions">
            <button className="icon-button" onClick={clearChat} aria-label="Clear conversation" title="Clear conversation">⌫</button>
            <button className="icon-button" onClick={() => setShowSettings(true)} aria-label="Connection settings" title="Connection settings">⚙</button>
          </div>
        </header>

        <div className="conversation" aria-live="polite">
          <div className="conversation-note">Today</div>
          {messages.map(message => <Message key={message.id} message={message} />)}
          {isLoading && <div className="message assistant loading"><span /><span /><span /></div>}
          {error && <p className="error-message">{error}</p>}
          <div ref={endRef} />
        </div>

        <form className="composer" onSubmit={sendMessage}>
          <label className="sr-only" htmlFor="chat-input">Your message</label>
          <textarea id="chat-input" value={draft} onChange={e => setDraft(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
            placeholder="Write a message…" rows="1" disabled={isLoading} />
          <button className="send-button" type="submit" disabled={!draft.trim() || isLoading} aria-label="Send message">↑</button>
        </form>
      </section>

      {showSettings && <div className="modal-backdrop" role="presentation" onMouseDown={() => setShowSettings(false)}>
        <section className="settings" role="dialog" aria-modal="true" aria-labelledby="settings-title" onMouseDown={e => e.stopPropagation()}>
          <h2 id="settings-title">Connection settings</h2>
          <p>Requests are sent as a POST to this URL.</p>
          <label htmlFor="endpoint">Chat endpoint</label>
          <input id="endpoint" value={endpoint} onChange={e => setEndpoint(e.target.value)} placeholder="https://api.example.com/chat" />
          <div className="settings-actions"><button className="secondary" onClick={() => setShowSettings(false)}>Close</button><button onClick={() => setShowSettings(false)}>Save</button></div>
        </section>
      </div>}
    </main>
  )
}

function Message({ message }) {
  const time = new Intl.DateTimeFormat([], { hour: 'numeric', minute: '2-digit' }).format(new Date(message.timestamp))
  return <article className={`message ${message.role}`}><p>{message.text}</p><time>{time}</time></article>
}

createRoot(document.getElementById('root')).render(<App />)
