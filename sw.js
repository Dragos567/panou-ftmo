// service worker minimal: doar cerinta PWA; datele sunt mereu proaspete din retea
self.addEventListener('install',e=>self.skipWaiting());
self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));
self.addEventListener('fetch',e=>{});
