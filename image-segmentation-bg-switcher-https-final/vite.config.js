import { defineConfig } from 'vite';
import fs from 'node:fs';

const useHttps = process.env.LOCAL_HTTPS === 'true';
const keyPath = './certs/localhost-key.pem';
const certPath = './certs/localhost-cert.pem';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    https: useHttps && fs.existsSync(keyPath) && fs.existsSync(certPath)
      ? { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }
      : false,
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    https: useHttps && fs.existsSync(keyPath) && fs.existsSync(certPath)
      ? { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }
      : false,
  },
});
