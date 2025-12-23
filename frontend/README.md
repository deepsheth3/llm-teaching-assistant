# 🎨 LLM Teaching Assistant - Frontend

A beautiful, modern React frontend for the LLM Teaching Assistant.

## ✨ Features

- 🌙 **Dark/Light Mode** - Automatic system detection + manual toggle
- 🎭 **Smooth Animations** - Framer Motion powered transitions
- 📱 **Fully Responsive** - Works on all devices
- ⚡ **Fast** - Vite-powered development
- 🎨 **Beautiful Design** - Tailwind CSS styling
- 📝 **Markdown Rendering** - Rich lesson display
- 🔄 **Real-time Streaming** - SSE support for lesson generation

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- Backend API running on http://localhost:8000

### Install

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open http://localhost:3000

### Build for Production

```bash
npm run build
npm run preview
```

## 📁 Project Structure

```
src/
├── components/          # React components
│   ├── Button.tsx      # Reusable button
│   ├── Card.tsx        # Card container
│   ├── Header.tsx      # Navigation header
│   ├── Hero.tsx        # Hero section with input
│   ├── Input.tsx       # Form inputs
│   ├── LessonDisplay.tsx   # Lesson modal
│   ├── Loading.tsx     # Loading states
│   └── ProblemDisplay.tsx  # LeetCode problem modal
├── hooks/
│   └── useTheme.tsx    # Theme context
├── lib/
│   ├── api.ts          # API client
│   └── utils.ts        # Utility functions
├── styles/
│   └── globals.css     # Global styles
├── App.tsx             # Main app component
└── main.tsx            # Entry point
```

## 🎨 Design System

### Colors

- **Primary**: Sky blue (#0ea5e9)
- **Accent**: Fuchsia (#d946ef)
- **Background**: White/Gray-950
- **Text**: Gray-900/Gray-100

### Typography

- **Sans**: Inter
- **Mono**: JetBrains Mono

## 🔧 Configuration

The app connects to the backend via Vite proxy. Configure in `vite.config.ts`:

```typescript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

## 📦 Dependencies

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Lucide React** - Icons
- **React Markdown** - Markdown rendering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License
