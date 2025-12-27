#!/bin/bash

# =============================================================================
# Complete Context Generator for Claude
# Captures ALL code files for full project understanding
# =============================================================================

OUTPUT_FILE="claude_context.md"

echo "# Complete Project Context for Claude" > $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Generated: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# -----------------------------------------------------------------------------
# 1. Project Structure
# -----------------------------------------------------------------------------
echo "## 1. Project Structure" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
tree -I 'node_modules|venv|__pycache__|.git|dist|build|*.pyc' --dirsfirst 2>/dev/null || find . -type f \( -name "*.py" -o -name "*.tsx" -o -name "*.ts" -o -name "*.json" -o -name "*.css" -o -name "*.md" \) ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/venv/*" ! -path "*/__pycache__/*" | head -100
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# -----------------------------------------------------------------------------
# 2. Backend - Core
# -----------------------------------------------------------------------------
echo "## 2. Backend Code" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "### 2.1 Core" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for file in backend/core/*.py; do
  if [ -f "$file" ]; then
    echo "#### $file" >> $OUTPUT_FILE
    echo '```python' >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
done

# -----------------------------------------------------------------------------
# 3. Backend - Models
# -----------------------------------------------------------------------------
echo "### 2.2 Models" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for file in backend/models/*.py; do
  if [ -f "$file" ]; then
    echo "#### $file" >> $OUTPUT_FILE
    echo '```python' >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
done

# -----------------------------------------------------------------------------
# 4. Backend - Services
# -----------------------------------------------------------------------------
echo "### 2.3 Services" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for file in backend/services/*.py; do
  if [ -f "$file" ]; then
    echo "#### $file" >> $OUTPUT_FILE
    echo '```python' >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
done

# -----------------------------------------------------------------------------
# 5. Backend - API
# -----------------------------------------------------------------------------
echo "### 2.4 API" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -f "backend/api/main.py" ]; then
  echo "#### backend/api/main.py" >> $OUTPUT_FILE
  echo '```python' >> $OUTPUT_FILE
  cat backend/api/main.py >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

for file in backend/api/routes/*.py; do
  if [ -f "$file" ]; then
    echo "#### $file" >> $OUTPUT_FILE
    echo '```python' >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
done

if [ -d "backend/api/middleware" ]; then
  for file in backend/api/middleware/*.py; do
    if [ -f "$file" ]; then
      echo "#### $file" >> $OUTPUT_FILE
      echo '```python' >> $OUTPUT_FILE
      cat "$file" >> $OUTPUT_FILE
      echo '```' >> $OUTPUT_FILE
      echo "" >> $OUTPUT_FILE
    fi
  done
fi

# -----------------------------------------------------------------------------
# 6. Backend - Requirements & Config
# -----------------------------------------------------------------------------
echo "### 2.5 Backend Config Files" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -f "backend/requirements.txt" ]; then
  echo "#### backend/requirements.txt" >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  cat backend/requirements.txt >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

if [ -f "backend/.env.example" ]; then
  echo "#### backend/.env.example" >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  cat backend/.env.example >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# 7. Frontend - Main Files
# -----------------------------------------------------------------------------
echo "## 3. Frontend Code" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "### 3.1 Entry Points" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -f "frontend/src/main.tsx" ]; then
  echo "#### frontend/src/main.tsx" >> $OUTPUT_FILE
  echo '```tsx' >> $OUTPUT_FILE
  cat frontend/src/main.tsx >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

if [ -f "frontend/src/App.tsx" ]; then
  echo "#### frontend/src/App.tsx" >> $OUTPUT_FILE
  echo '```tsx' >> $OUTPUT_FILE
  cat frontend/src/App.tsx >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# 8. Frontend - Components
# -----------------------------------------------------------------------------
echo "### 3.2 Components" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for file in frontend/src/components/*.tsx; do
  if [ -f "$file" ]; then
    echo "#### $file" >> $OUTPUT_FILE
    echo '```tsx' >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
done

# Also check for nested component folders
if [ -d "frontend/src/components/ui" ]; then
  for file in frontend/src/components/ui/*.tsx; do
    if [ -f "$file" ]; then
      echo "#### $file" >> $OUTPUT_FILE
      echo '```tsx' >> $OUTPUT_FILE
      cat "$file" >> $OUTPUT_FILE
      echo '```' >> $OUTPUT_FILE
      echo "" >> $OUTPUT_FILE
    fi
  done
fi

# -----------------------------------------------------------------------------
# 9. Frontend - Lib/Utils
# -----------------------------------------------------------------------------
echo "### 3.3 Lib & Utils" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -d "frontend/src/lib" ]; then
  for file in frontend/src/lib/*.ts frontend/src/lib/*.tsx; do
    if [ -f "$file" ]; then
      echo "#### $file" >> $OUTPUT_FILE
      echo '```typescript' >> $OUTPUT_FILE
      cat "$file" >> $OUTPUT_FILE
      echo '```' >> $OUTPUT_FILE
      echo "" >> $OUTPUT_FILE
    fi
  done
fi

if [ -d "frontend/src/utils" ]; then
  for file in frontend/src/utils/*.ts frontend/src/utils/*.tsx; do
    if [ -f "$file" ]; then
      echo "#### $file" >> $OUTPUT_FILE
      echo '```typescript' >> $OUTPUT_FILE
      cat "$file" >> $OUTPUT_FILE
      echo '```' >> $OUTPUT_FILE
      echo "" >> $OUTPUT_FILE
    fi
  done
fi

# -----------------------------------------------------------------------------
# 10. Frontend - Styles
# -----------------------------------------------------------------------------
echo "### 3.4 Styles" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -f "frontend/src/index.css" ]; then
  echo "#### frontend/src/index.css" >> $OUTPUT_FILE
  echo '```css' >> $OUTPUT_FILE
  cat frontend/src/index.css >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# 11. Frontend - Config Files
# -----------------------------------------------------------------------------
echo "### 3.5 Frontend Config Files" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -f "frontend/package.json" ]; then
  echo "#### frontend/package.json" >> $OUTPUT_FILE
  echo '```json' >> $OUTPUT_FILE
  cat frontend/package.json >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

if [ -f "frontend/vite.config.ts" ]; then
  echo "#### frontend/vite.config.ts" >> $OUTPUT_FILE
  echo '```typescript' >> $OUTPUT_FILE
  cat frontend/vite.config.ts >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

if [ -f "frontend/tailwind.config.js" ]; then
  echo "#### frontend/tailwind.config.js" >> $OUTPUT_FILE
  echo '```javascript' >> $OUTPUT_FILE
  cat frontend/tailwind.config.js >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

if [ -f "frontend/tsconfig.json" ]; then
  echo "#### frontend/tsconfig.json" >> $OUTPUT_FILE
  echo '```json' >> $OUTPUT_FILE
  cat frontend/tsconfig.json >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# 12. Root Config Files
# -----------------------------------------------------------------------------
echo "## 4. Root Config Files" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -f "README.md" ]; then
  echo "#### README.md" >> $OUTPUT_FILE
  echo '```markdown' >> $OUTPUT_FILE
  cat README.md >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

if [ -f ".gitignore" ]; then
  echo "#### .gitignore" >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  cat .gitignore >> $OUTPUT_FILE
  echo '```' >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# 13. Railway Config (if exists)
# -----------------------------------------------------------------------------
if [ -f "railway.json" ] || [ -f "railway.toml" ]; then
  echo "## 5. Railway Deployment Config" >> $OUTPUT_FILE
  echo "" >> $OUTPUT_FILE
  
  if [ -f "railway.json" ]; then
    echo "#### railway.json" >> $OUTPUT_FILE
    echo '```json' >> $OUTPUT_FILE
    cat railway.json >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
  
  if [ -f "railway.toml" ]; then
    echo "#### railway.toml" >> $OUTPUT_FILE
    echo '```toml' >> $OUTPUT_FILE
    cat railway.toml >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
  fi
fi

# -----------------------------------------------------------------------------
# 14. Summary
# -----------------------------------------------------------------------------
echo "## 6. Project Summary" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "### Current v2 Features" >> $OUTPUT_FILE
echo "- ✅ Relevance thresholds (0.50/0.35/0.20)" >> $OUTPUT_FILE
echo "- ✅ Query enhancement (intent detection)" >> $OUTPUT_FILE
echo "- ✅ Dynamic paper fetching (Semantic Scholar)" >> $OUTPUT_FILE
echo "- ✅ LeetCode removed" >> $OUTPUT_FILE
echo "- ✅ FAISS vector search" >> $OUTPUT_FILE
echo "- ✅ GROBID PDF parsing" >> $OUTPUT_FILE
echo "- ✅ GPT-4o-mini lesson generation" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "### TODO for Next Phase" >> $OUTPUT_FILE
echo "- 🔜 Migrate FAISS → Pinecone (persistent dynamic updates)" >> $OUTPUT_FILE
echo "- 🔜 Multi-paper comparison lessons" >> $OUTPUT_FILE
echo "- 🔜 Concept Map UI" >> $OUTPUT_FILE
echo "- 🔜 User feedback system" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo "---" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "*Context generated for Claude to understand the complete codebase*" >> $OUTPUT_FILE

echo ""
echo "✅ Context generated: $OUTPUT_FILE"
echo "📊 Lines: $(wc -l < $OUTPUT_FILE)"
echo "📦 Size: $(du -h $OUTPUT_FILE | cut -f1)"
echo ""
echo "Upload this file to Claude!"