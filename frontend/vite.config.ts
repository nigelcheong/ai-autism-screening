import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    // Dev-only: lets a tunnel's (randomly-generated) hostname through
    // Vite's Host-header check, since it can't be known in advance.
    allowedHosts: true,
    proxy: {
      "/internal": {
        target: "http://localhost:8010",
        changeOrigin: true,
      },
    },
  },
});
