# Verdant Chat frontend

React/Vite chatbot interface. It sends `POST` requests to `/chat` (or a configured full URL) with `{ "message": "..." }`.

## Configure the backend

Use the settings button in the interface to enter the live endpoint; it is retained in browser storage. Or copy `.env.example` to `.env` and set `VITE_CHAT_ENDPOINT` before starting the app.

## Run

```bash
npm install
npm run dev
```

The response adapter is deliberately located in `src/main.jsx` as `getResponseText()`, ready to align with the eventual backend JSON shape.
