import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
    build: {
        outDir: "build"
    },
    server: {
        port: 3000,
        host: 'localhost',
        fs: {
            strict: false,
            allow: ['..']
        }
    },
    test: {
        globals: true,
        environment: "jsdom",
        setupFiles: "./src/setupTests.js"
    },
});
