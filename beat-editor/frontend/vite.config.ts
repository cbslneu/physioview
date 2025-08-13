import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
    build: {
        outDir: "build"
    },
    server: {
        port: 3000,
        host: '0.0.0.0',
        watch: {
            usePolling: true
        },
        fs: {
            strict: false,
            allow: ['..']
        }
    }
});
