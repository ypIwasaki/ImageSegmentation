import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

fs.mkdirSync('certs', { recursive: true });
const key = 'certs/localhost-key.pem';
const cert = 'certs/localhost-cert.pem';

if (fs.existsSync(key) && fs.existsSync(cert)) {
  process.exit(0);
}

try {
  execFileSync('openssl', [
    'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
    '-keyout', key,
    '-out', cert,
    '-days', '3650',
    '-subj', '/CN=localhost',
    '-addext', 'subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1',
  ], { stdio: 'inherit' });
  console.log('Generated self-signed HTTPS certificate for localhost.');
} catch (error) {
  console.error('Failed to generate certificate. Install openssl or use mkcert.');
  process.exit(1);
}
