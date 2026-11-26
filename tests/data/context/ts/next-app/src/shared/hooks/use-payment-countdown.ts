import { useEffect, useState } from 'react';

function formatCountdown(seconds: number): string {
  const m = Math.floor(seconds / 60);
  return `${m}:${String(seconds % 60).padStart(2, '0')}`;
}

export function usePaymentCountdown(expiresAt: number | undefined) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const secondsLeft = expiresAt ? Math.max(0, Math.floor((expiresAt - now) / 1000)) : 0;
  return { countdown: formatCountdown(secondsLeft), secondsLeft };
}

/**
 * Whether `expiresAt` (epoch milliseconds) has passed. It flips once, on a timer,
 * instead of on every tick.
 */
export function useIsExpired(expiresAt: number | undefined) {
  const [expired, setExpired] = useState(false);
  useEffect(() => {
    if (expiresAt == null) return;
    const id = setTimeout(() => setExpired(true), expiresAt - Date.now());
    return () => clearTimeout(id);
  }, [expiresAt]);
  return expired;
}
