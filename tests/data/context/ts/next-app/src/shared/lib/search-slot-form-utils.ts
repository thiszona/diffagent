import { computeEarliestStart } from '@/lib/time';

export function earliestToday(now: Date) {
  const first = computeEarliestStart(now, 4);
  const again = computeEarliestStart(now, 4);
  return first ?? again;
}

export function earliestTomorrow(now: Date) {
  const next = new Date(now.getTime() + 24 * 60 * 60 * 1000);
  return computeEarliestStart(next, 4);
}

export function earliestLater(now: Date) {
  return computeEarliestStart(now, 8);
}
