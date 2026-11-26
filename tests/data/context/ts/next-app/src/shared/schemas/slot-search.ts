import * as time from '@/lib/time';

export function slotSearchEarliest(now: Date) {
  return time.computeEarliestStart(now, 4);
}
