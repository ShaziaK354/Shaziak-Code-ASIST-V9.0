// vite.config.js
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    proxy: {
      // All API requests (including /api/auth/login, /api/auth/callback, /api/auth/logout)
      '/api': {
        target: 'http://localhost:3000', // Your Flask backend
        changeOrigin: true,
      }
      // NOTE: /login, /callback, /logout proxy entries REMOVED
      // Auth routes are now under /api/auth/* so they go through the /api proxy
    }
  }
})
