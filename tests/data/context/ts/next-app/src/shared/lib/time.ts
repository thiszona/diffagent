/** Hours and minutes of `now` in the server's time zone. */
export function getServerHM(now: Date, tz?: string): { h: number; m: number } {
  const d = tz ? new Date(now.toLocaleString('en-US', { timeZone: tz })) : now;
  return { h: d.getHours(), m: d.getMinutes() };
}

/** Rounds minutes up to the next 30-minute slot, or undefined past 23:59. */
export function roundUpTo30Min(totalMinutes: number): { h: number; m: number } | undefined {
  const rounded = Math.ceil(totalMinutes / 30) * 30;
  if (rounded >= 24 * 60) return undefined;
  return { h: Math.floor(rounded / 60), m: rounded % 60 };
}

/**
 * Computes the earliest allowed start time: now plus a lead time in
 * minutes, rounded up to the next 30-minute slot.
 */
export function computeEarliestStart(
  now: Date,
  advanceMinutes = 4 * 60,
  tz?: string
): { h: number; m: number } | undefined {
  const { h, m } = getServerHM(now, tz);
  return roundUpTo30Min(h * 60 + m + advanceMinutes);
}

/** Formats hours and minutes as HH:MM. */
export function formatHM(h: number, m: number): string {
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}
