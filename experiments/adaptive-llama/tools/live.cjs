// Read-only, loopback-only viewer. No Python, package install, training controls or autostart.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(process.argv[2] || 'artifacts/adaptive-llama/continual-01');
const port = Number(process.argv[3] || 8090);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('Invalid port');
const template = fs.readFileSync(path.join(__dirname, '../ui/campaign.html'), 'utf8').replace('__REPORT_JSON__', 'null');
const server = http.createServer((req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  const host = req.headers.host;
  if (host !== `127.0.0.1:${port}` && host !== `localhost:${port}`) { res.writeHead(403); res.end(); return; }
  if (req.method !== 'GET') { res.writeHead(405); res.end(); return; }
  if (req.url === '/' || req.url === '/index.html') {
    res.setHeader('Content-Type', 'text/html; charset=utf-8'); res.end(template);
  } else if (req.url === '/status.json') {
    fs.readFile(path.join(directory, 'status.json'), (err, data) => {
      if (err) { res.writeHead(err.code === 'ENOENT' ? 404 : 500); res.end('No readable campaign status yet.'); return; }
      res.setHeader('Content-Type', 'application/json'); res.end(data);
    });
  } else { res.writeHead(404); res.end(); }
});
server.on('error', e => { console.error(e.message); process.exitCode = 1; });
server.listen(port, '127.0.0.1', () => console.log(`Live view: http://127.0.0.1:${port}/\nRun: ${directory}\nCtrl+C stops only this viewer.`));
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close());
