import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Backend runs on :8000 locally (see Dockerfile / README). In
      // production the two are served from different domains and
      // VITE_API_BASE_URL (src/lib/api.ts) is set to the real API URL
      // instead - this proxy is a local-dev convenience only.
      "/api/v1": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
