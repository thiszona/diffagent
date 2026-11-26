/** Drops the stored registration session. */
export function clearAuthSession() {
  sessionStorage.removeItem('register');
}

export function getRegisterSession(): { expired_at: number } | null {
  const raw = sessionStorage.getItem('register');
  return raw ? JSON.parse(raw) : null;
}
