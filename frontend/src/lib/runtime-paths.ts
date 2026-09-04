/** Runtime URLs that work both at the local root and under a Pages subpath. */
const configuredApiBaseUrl = (import.meta.env?.VITE_API_BASE_URL ?? '').replace(
  /\/+$/,
  '',
);

export const apiUrl = (path: string) =>
  `${configuredApiBaseUrl}${path.startsWith('/') ? path : `/${path}`}`;

const baseUrl = import.meta.env?.BASE_URL || '/';
const normalizedBaseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;

export const publicAssetUrl = (path: string) =>
  `${normalizedBaseUrl}${path.replace(/^\/+/, '')}`;

export const routerBasename =
  normalizedBaseUrl === '/' ? '/' : normalizedBaseUrl;
